import os
import re
import subprocess
import tempfile

import editdistance
import numpy as np
from six.moves import urllib


#######################################################################################################################


class Scorer(object):
    """
    Calculation of BLEU/WER/CER between word sequence (hypothesis) and word sequence (reference).
    """

    def __init__(self, reduction="sum"):
        self.reduction = reduction

    def __call__(self, preds, labels):
        return self.get_wer_cer(preds, labels, reduction=self.reduction)[0]

    @staticmethod
    def get_wer_cer(hypotheses, references, reduction="mean"):
        """
        Mean normalised edit distance between predictions and labels
        ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf  - Chapter 2.1. Label Error Rate
        """

        assert reduction in ['mean', 'sum'], 'Reduction parameter wrong.'

        idx, total_cer, total_wer = 0, 0.0, 0.0
        for idx, (reference, hypothesis) in enumerate(zip(references, hypotheses)):
            hypothesis_words = list(filter(None, hypothesis.split(' ')))
            reference_words = list(filter(None, reference.split(' ')))

            assert len(reference_words) > 0, "Reference word count must be greater 0"

            total_wer += (editdistance.eval(hypothesis_words, reference_words) / len(reference_words))

            hypothesis_chars = ''.join(hypothesis_words)  # exclude spaces from calculation of edit disance
            reference_chars = ''.join(reference_words)

            assert len(reference_chars) > 0, "Reference char count must be greater 0"

            total_cer += (editdistance.eval(hypothesis_chars, reference_chars) / len(reference_chars))

        if 'mean' in reduction:
            total_wer /= (idx + 1)
            total_cer /= (idx + 1)

        return total_wer, total_cer

    @staticmethod
    def get_moses_multi_bleu(hypotheses, references, lowercase=False):
        """
        From https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.metrics.html
        modified to look for perl script locally first

        """
        if isinstance(hypotheses, list):
            hypotheses = np.array(hypotheses)
        if isinstance(references, list):
            references = np.array(references)

        if np.size(hypotheses) == 0:
            return np.float32(0.0)

        multi_bleu_path = "./multi-bleu.perl"
        if not os.path.exists(multi_bleu_path):
            # Get MOSES multi-bleu script
            try:
                multi_bleu_path, _ = urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
                    "master/scripts/generic/multi-bleu.perl")
                os.chmod(multi_bleu_path, 0o755)
            except:
                print("Unable to fetch multi-bleu.perl script")
                return None

        # Dump hypotheses and references to tempfiles
        hypothesis_file = tempfile.NamedTemporaryFile()
        hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
        hypothesis_file.write(b"\n")
        hypothesis_file.flush()
        reference_file = tempfile.NamedTemporaryFile()
        reference_file.write("\n".join(references).encode("utf-8"))
        reference_file.write(b"\n")
        reference_file.flush()

        # Calculate BLEU using multi-bleu script
        with open(hypothesis_file.name, "r") as read_pred:
            bleu_cmd = [multi_bleu_path]
            if lowercase:
                bleu_cmd += ["-lc"]
            bleu_cmd += [reference_file.name]
            try:
                bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
                bleu_out = bleu_out.decode("utf-8")
                bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
                bleu_score = float(bleu_score)
                bleu_score = np.float32(bleu_score)
            except subprocess.CalledProcessError as error:
                if error.output is not None:
                    print("multi-bleu.perl script returned non-zero exit code")
                    print(error.output)
                bleu_score = None

        # Close temp files
        hypothesis_file.close()
        reference_file.close()

        return bleu_score

    @staticmethod
    def get_acc(hypotheses, references, reduction="mean"):

        assert reduction in ['mean', 'none'], 'Reduction parameter wrong.'

        lines = 0
        correct_lines = []
        for dseq, tseq in zip(hypotheses, references):
            lines += 1
            if not sum([d != t for d, t in zip(dseq, tseq)]):
                correct_lines.append((dseq, tseq))

        if 'none' in reduction:
            return correct_lines
        else:
            return len(correct_lines) / float(lines)

#######################################################################################################################
