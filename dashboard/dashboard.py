import json
from glob import glob

import torch
import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf, DictConfig
from streamlit_pianoroll import from_fortepyan

from data.dataset import MidiDataset
from models.mae import MidiMaskedAutoencoder

device = "cpu"


def display_pianoroll(processing_result: dict):
    st.json(processing_result["source"], expanded=False)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Original")
        from_fortepyan(processing_result["original"])
        fig = ff.view.draw_dual_pianoroll(processing_result["original"])
        st.pyplot(fig)

    with col2:
        st.write("### Generated")
        from_fortepyan(processing_result["generated"])
        fig = ff.view.draw_dual_pianoroll(processing_result["generated"])
        st.pyplot(fig)


def denormalize_velocity(velocity: np.ndarray):
    return ((velocity + 1) * 64).astype("int")


def denormalize_time_features(time_feature: np.ndarray, mean: float, std: float):
    time_feature = std * time_feature + mean
    return 2**time_feature - 1e-8


def to_midi_piece(
    pitch: np.ndarray,
    dstart: np.ndarray,
    duration: np.ndarray,
    velocity: np.ndarray,
    mask: np.ndarray = None,
) -> ff.MidiPiece:
    record = {
        "pitch": pitch,
        "velocity": velocity,
        "dstart": dstart.astype("float"),
        "duration": duration.astype("float"),
        "mask": mask,
    }

    df = pd.DataFrame(record)
    df["start"] = df.dstart.cumsum().shift(1).fillna(0)
    df["end"] = df.start + df.duration

    return ff.MidiPiece(df)


# @st.cache_data
def get_model(checkpoint_path: str) -> MidiMaskedAutoencoder:
    checkpoint = torch.load(checkpoint_path)

    cfg = checkpoint["config"]
    st.json(OmegaConf.to_container(cfg), expanded=False)

    model = MidiMaskedAutoencoder(
        encoder_dim=cfg.model.encoder_dim,
        encoder_depth=cfg.model.encoder_depth,
        encoder_num_heads=cfg.model.encoder_num_heads,
        decoder_dim=cfg.model.decoder_dim,
        decoder_depth=cfg.model.decoder_depth,
        decoder_num_heads=cfg.model.decoder_num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dynamics_embedding_depth=cfg.model.dynamics_embedding_depth,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, cfg


def model_selection() -> MidiMaskedAutoencoder:
    checkpoints = glob("checkpoints/*")
    checkpoint_path = st.selectbox("Select checkpoint", options=checkpoints)

    model, cfg = get_model(checkpoint_path)
    return model, cfg


@st.cache_data
def get_dataset() -> Dataset:
    dataset_name = "JasiekKaczmarczyk/maestro-v1-sustain-masked"
    dataset = load_dataset(dataset_name, split="validation")
    return dataset


def main():
    # Load model
    model, cfg = model_selection()

    # Prep data
    dataset = get_dataset()
    source = [json.loads(source) for source in dataset["source"]]
    source_df = pd.DataFrame(source)

    composers = source_df.composer.unique()
    selected_composer = st.selectbox(
        label="Select composer",
        options=composers,
        index=3,
    )

    ids = source_df.composer == selected_composer
    piece_titles = source_df[ids].title.unique()
    selected_title = st.selectbox(
        label="Select title",
        options=piece_titles,
    )
    st.write(selected_title)

    ids = (source_df.composer == selected_composer) & (source_df.title == selected_title)
    n_samples = 10
    seed = 137
    part_df = source_df[ids].sample(n_samples, random_state=seed)

    idxs = part_df.index.values
    part_dataset = dataset.select(idxs)
    midi_dataset = MidiDataset(part_dataset, use_dstart_log_normalization=cfg.train.use_dstart_log_normalization)

    if cfg.train.use_dstart_log_normalization:
        mean_dstart = midi_dataset.mean_dstart
        std_dstart = midi_dataset.std_dstart
    else:
        mean_dstart = None
        std_dstart = None

    masking_ratio = st.number_input(
        label="Masking ratio",
        min_value=0.05,
        max_value=1.0,
        value=0.5,
    )
    generated_pieces = generate_pieces(
        midi_dataset=midi_dataset,
        model=model,
        masking_ratio=masking_ratio,
        mean_dstart=mean_dstart,
        std_dstart=std_dstart,
    )

    for processing_result in generated_pieces:
        display_pianoroll(processing_result)


@torch.no_grad()
def generate_pieces(
    midi_dataset: MidiDataset,
    model: MidiMaskedAutoencoder,
    masking_ratio: float,
    mean_dstart: float = None,
    std_dstart: float = None,
):
    dataloader = DataLoader(midi_dataset, batch_size=256, shuffle=True)

    generated_pieces = []
    for batch in dataloader:
        pitches = batch["pitch"].to(device)
        velocities = batch["velocity"].to(device)
        dstarts = batch["dstart"].to(device)
        durations = batch["duration"].to(device)

        pred_pitches, pred_velocities, pred_dstarts, pred_durations, mask = model(
            pitch=pitches,
            velocity=velocities,
            dstart=dstarts,
            duration=durations,
            masking_ratio=masking_ratio,
        )

        # replace tokens that were masked with generated values
        mask = mask.detach().bool()
        pred_pitches = torch.argmax(pred_pitches, dim=-1)

        gen_pitches = torch.where(mask, pred_pitches, pitches)
        gen_velocities = torch.where(mask, pred_velocities, velocities)
        gen_dstarts = torch.where(mask, pred_dstarts, dstarts)
        gen_durations = torch.where(mask, pred_durations, durations)

        generated_pieces += decode_batch(
            batch=batch,
            gen_pitches=gen_pitches,
            gen_velocities=gen_velocities,
            gen_dstarts=gen_dstarts,
            gen_durations=gen_durations,
            mask=mask,
            mean_dstart=mean_dstart,
            std_dstart=std_dstart,
        )

    return generated_pieces


def decode_batch(
    batch: torch.Tensor,
    gen_pitches: torch.Tensor,
    gen_velocities: torch.Tensor,
    gen_dstarts: torch.Tensor,
    gen_durations: torch.Tensor,
    mask: torch.Tensor,
    mean_dstart: float = None,
    std_dstart: float = None,
):
    sources = batch["source"]
    pitches = batch["pitch"].to(device)
    velocities = batch["velocity"].to(device)
    dstarts = batch["dstart"].to(device)
    durations = batch["duration"].to(device)

    batch_size = pitches.shape[0]

    generated_pieces = []
    for it in range(batch_size):
        pitch = pitches[it].cpu().numpy() + 21
        velocity = velocities[it].cpu().numpy()
        dstart = dstarts[it].cpu().numpy()
        duration = durations[it].cpu().numpy()

        gen_pitch = gen_pitches[it].cpu().numpy() + 21
        gen_velocity = gen_velocities[it].cpu().numpy()
        gen_dstart = gen_dstarts[it].cpu().numpy()
        gen_duration = gen_durations[it].cpu().numpy()

        m = mask[it].cpu().numpy()

        velocity = denormalize_velocity(velocity)
        gen_velocity = denormalize_velocity(gen_velocity)

        if mean_dstart is not None and std_dstart is not None:
            dstart = denormalize_time_features(dstart, mean=mean_dstart, std=std_dstart)
            gen_dstart = denormalize_time_features(gen_dstart, mean=mean_dstart, std=std_dstart)

        original_piece = to_midi_piece(pitch, dstart, duration, velocity, mask=m)
        model_piece = to_midi_piece(gen_pitch, gen_dstart, gen_duration, gen_velocity, mask=m)

        processing_result = {
            "original": original_piece,
            "generated": model_piece,
            "source": json.loads(sources[it]),
        }
        generated_pieces.append(processing_result)

    return generated_pieces


if __name__ == "__main__":
    main()
