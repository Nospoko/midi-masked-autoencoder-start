import os
import uuid
import pickle
import argparse

import torch
import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
from datasets import load_dataset
from torch.utils.data import Subset, DataLoader

from data.dataset import MidiDataset
from models.mae import MidiMaskedAutoencoder


def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def preprocess_dataset(
    index: int,
):
    dataset = load_dataset("roszcz/maestro-v1-sustain", split="validation")
    record = dataset[index]

    piece = ff.MidiPiece.from_huggingface(record)

    piece.df["next_start"] = piece.df.start.shift(-1)
    piece.df["dstart"] = piece.df.next_start - piece.df.start
    piece.df["dstart"] = piece.df["dstart"].fillna(0)

    midi_filename = piece.source["midi_filename"]

    # sanity check, replace NaN with 0
    if np.any(np.isnan(piece.df["dstart"])):
        piece.df["dstart"] = np.nan_to_num(piece.df["dstart"], copy=False)

    data = {
        "filename": midi_filename,
        "pitch": torch.tensor(piece.df["pitch"], dtype=torch.long) - 21,
        "velocity": (torch.tensor(piece.df["velocity"], dtype=torch.float) / 64) - 1,
        "dstart": torch.tensor(piece.df["dstart"], dtype=torch.float).clip(0.0, 5.0),
        "duration": torch.tensor(piece.df["duration"], dtype=torch.float).clip(0.0, 5.0),
    }

    return data


def denormalize_velocity(velocity: np.ndarray):
    return ((velocity + 1) * 64).astype("int")


def to_midi_piece(
    pitch: np.ndarray, dstart: np.ndarray, duration: np.ndarray, velocity: np.ndarray, mask: np.ndarray = None
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


def process_file(
    model: MidiMaskedAutoencoder,
    index: int,
    save_path: str,
    device: torch.device,
):
    batch = preprocess_dataset(index)

    # dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    processed_midi_pieces = {}

    with torch.no_grad():
        filenames = batch["filename"]
        pitches = batch["pitch"][None, :].to(device)
        velocities = batch["velocity"][None, :].to(device)
        dstarts = batch["dstart"][None, :].to(device)
        durations = batch["duration"][None, :].to(device)

        pred_pitches, pred_velocities, pred_dstarts, pred_durations, mask = model(
            pitch=pitches,
            velocity=velocities,
            dstart=dstarts,
            duration=durations,
            masking_ratio=0.5,
        )
        mask = mask.detach().bool()
        pred_pitches = torch.argmax(pred_pitches, dim=-1)

        # gen_pitches = pred_pitches
        # gen_velocities = pred_velocities
        # gen_dstarts = pred_dstarts
        # gen_durations = pred_durations

        # replace tokens that were masked with generated values
        gen_pitches = torch.where(mask, pred_pitches, pitches)
        gen_velocities = torch.where(mask, pred_velocities, velocities)
        gen_dstarts = torch.where(mask, pred_dstarts, dstarts)
        gen_durations = torch.where(mask, pred_durations, durations)

        for i in range(len(pitches)):
            name = f"{filenames[i]}-{str(uuid.uuid1())[:8]}"

            pitch = pitches[i].cpu().numpy() + 21
            velocity = velocities[i].cpu().numpy()
            dstart = dstarts[i].cpu().numpy()
            duration = durations[i].cpu().numpy()
            gen_pitch = gen_pitches[i].cpu().numpy() + 21
            gen_velocity = gen_velocities[i].cpu().numpy()
            gen_dstart = gen_dstarts[i].cpu().numpy()
            gen_duration = gen_durations[i].cpu().numpy()

            m = mask[i].cpu().numpy()

            velocity = denormalize_velocity(velocity)
            gen_velocity = denormalize_velocity(gen_velocity)

            original_piece = to_midi_piece(pitch, dstart, duration, velocity, mask=m)
            model_piece = to_midi_piece(gen_pitch, gen_dstart, gen_duration, gen_velocity, mask=m)

            processed_midi_pieces[name] = {
                "original": original_piece,
                "generated": model_piece,
            }

            # original_midi = original_piece.to_midi()
            # model_midi = model_piece.to_midi()

            # # save as midi
            # original_midi.write(f"{save_path}/original/{query}-{i}-original.midi")
            # model_midi.write(f"{save_path}/generated/{query}-{i}-model.midi")

    with open(save_path, "wb") as handle:
        pickle.dump(processed_midi_pieces, handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    checkpoint = torch.load("checkpoints/mae10m-2023-11-09-10-35-params-9.88M.ckpt")

    cfg = checkpoint["config"]
    # device = cfg.train.device
    device = "cpu"

    model = MidiMaskedAutoencoder(
        encoder_dim=cfg.model.encoder_dim,
        encoder_depth=cfg.model.encoder_depth,
        encoder_num_heads=cfg.model.encoder_num_heads,
        decoder_dim=cfg.model.decoder_dim,
        decoder_depth=cfg.model.decoder_depth,
        decoder_num_heads=cfg.model.decoder_num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    # dataset_name = "JasiekKaczmarczyk/maestro-v1-sustain-masked"
    save_path = "tmp/processed_long_file.pickle"

    # makedir_if_not_exists(f"{save_path}/generated")
    # makedir_if_not_exists(f"{save_path}/original")

    process_file(model, index=105, save_path=save_path, device=device)


if __name__ == "__main__":
    main()
