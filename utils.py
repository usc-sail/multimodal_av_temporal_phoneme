from torch.utils.data import DataLoader
from dataloader import VideoAudioPhonemeDataset
import torch
from Levenshtein import distance as levenshtein_distance
import json

def get_dataset(video_directory, batch_size, modality="f"):
    train_set = VideoAudioPhonemeDataset(video_directory, training=True, modality=modality)
    test_set = VideoAudioPhonemeDataset(video_directory, training=False, modality=modality)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader


class PhonemeErrorRate:
    def __init__(self, phoneme_vocab, blank_id=0):
        """
        Args:
            phoneme_vocab (List[str]): ID-to-phoneme mapping
            blank_id (int): ID of the CTC blank token
        """
        self.phoneme_vocab = phoneme_vocab
        self.blank_id = blank_id
        self.total_edits = 0
        self.total_ref_phonemes = 0
        self.decoded_refs = []
        self.decoded_hyps = []
    
    def reset(self):
        """
        Reset the internal state of the metric.
        """
        self.total_edits = 0
        self.total_ref_phonemes = 0
        self.decoded_refs = []
        self.decoded_hyps = []

    def greedy_ctc_decode(self, log_probs, prob=True):
        """
        Greedy CTC decoding from log_probs to phoneme sequence.
        
        Args:
            log_probs (Tensor): shape [T, V]
            prob (bool): If True, assumes input is log_probs from model output (hyp);
                         If False, assumes manual or clean target (ref)
        
        Returns:
            List[str]: Decoded phoneme sequence
        """
        if prob:
            pred_ids = torch.argmax(log_probs, dim=-1).tolist()
        else:
            # assert False, log_probs.shape
            pred_ids = log_probs.tolist()
        seq = []
        prev = None
        for idx in pred_ids:
            if idx == self.blank_id:
                continue
            if idx != prev:
                seq.append(self.phoneme_vocab[idx])
            prev = idx
        return seq

    def add_batch(self, log_probs_ref_batch, log_probs_hyp_batch):
        """
        Add a batch of phoneme predictions and references.
        
        Args:
            log_probs_ref_batch: List[Tensor] — reference log-probs per sample [T_i, V]
            log_probs_hyp_batch: List[Tensor] — hypothesis log-probs per sample [T_i, V]
        """
        for log_probs_ref, log_probs_hyp in zip(log_probs_ref_batch, log_probs_hyp_batch):
            ref_seq = self.greedy_ctc_decode(log_probs_ref, prob=True)
            hyp_seq = self.greedy_ctc_decode(log_probs_hyp, prob=False)

            self.decoded_refs.append(ref_seq)
            self.decoded_hyps.append(hyp_seq)

            dist = levenshtein_distance(ref_seq, hyp_seq)

            self.total_edits += dist
            self.total_ref_phonemes += len(ref_seq)

    def compute(self):
        """
        Returns:
            per (float): Micro-average PER over all batches
            total_edits (int): Total Levenshtein distance
            total_ref_phonemes (int): Total number of reference phonemes
        """
        per = self.total_edits / max(self.total_ref_phonemes, 1)
        return per, self.total_edits, self.total_ref_phonemes
    
    def save(self, path):
        """
        Save ref-hyp pairs to a JSON file.

        Each entry in the JSON file is a dict:
        { "reference": ..., "hypothesis": ... }
        """
        data = [
            {"pred": ref, "true": hyp}
            for ref, hyp in zip(self.decoded_refs, self.decoded_hyps)
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

