
<b>Neural Speech Recognition</b>

This repository contains experiments with CNN, Deepspeech2 and RNN models with different datasets.

The follwing results have been created with the <a href="http://www.speech.cs.cmu.edu/databases/an4/">AN4 dataset</a>:

![Alt text](resources/validscore.jpg?raw=true "")

![Alt text](resources/validloss.jpg?raw=true "")

(orange:ResNet, blue:ResNet+augmentation, dark red: Deepspeech, light blue: Deepspeech+augmentation, ligth red: EncoderDecoder,
green: EncoderDecoder+augmentation, grap: EncoderDecoder+augmentation+pseudo labels)

ResNet CNN + CTC

Bleu: 70.180 WER:  16.482 CER:  9.792 ACC:  49.231

Deepspeech 2 + CTC

Bleu: 89.890 WER:  5.094 CER:  2.993 ACC:  76.923

Encoder Decoder RNN 

Bleu: 83.770 WER:  7.529 CER:  5.735 ACC:  72.308
