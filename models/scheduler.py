class MaskingRatioScheduler:
    def __init__(self, masking_ratio_mapping: dict):
        self.masking_ratio_mapping = masking_ratio_mapping

    def get_masking_ratio(self, num_tokens_processed: int):
        masking_ratio = max(
            [self.masking_ratio_mapping[key] for key in self.masking_ratio_mapping if key <= num_tokens_processed]
        )

        return masking_ratio
