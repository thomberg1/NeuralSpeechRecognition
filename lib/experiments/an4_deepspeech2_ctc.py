import random

from torch import optim

from lib.checkpoint import *
from lib.dataloader.audio import AudioDataset, collate_fn
from lib.datasets.an4 import create_data_pipelines
from lib.decoder import CTCGreedyDecoder
from lib.evaluator import Evaluator
from lib.loss import PytorchCTCLoss
from lib.models.deepspeech2 import DeepSpeech2
from lib.recognizer import Recognizer
from lib.scorer import Scorer
from lib.stopping import Stopping
from lib.tools import *
from lib.trainer import Trainer
from lib.trainlogger import *
from lib.transforms.audio import AudioSpectrogram, AudioNormalizeDB, AudioNormalize
from lib.transforms.general import FromNumpyToTensor, TranscriptEncodeCTC
from lib.utilities import *
from ..vocabulary import Vocabulary

logger = logging.getLogger(__name__)


#######################################################################################################################


def run_training(H):
    # torch.cuda.is_available = lambda : False
    # torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    create_logger(H)

    random.seed(H.SEED)
    np.random.seed(H.SEED)
    torch.manual_seed(H.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(H.SEED)
        torch.cuda.manual_seed_all(H.SEED)

    logger.info("Training start.")
    logger.info(repr(H))

    train_loader, valid_loader, vocab = create_data_pipelines(H)

    logger.info(train_loader.dataset)
    logger.info(valid_loader.dataset)

    m = Metric([('train_loss', np.inf), ('train_score', np.inf), ('valid_loss', np.inf), ('valid_score', 0),
                ('train_lr', 0), ('valid_cer', np.inf)])

    model = DeepSpeech2(len(vocab), rnn_hidden_size=H.RNN_HIDDEN_SIZE, nb_layers=H.NUM_LAYERS,
                        bidirectional=H.BIDIRECTIONAL, cnn_dropout=H.CNN_DROPOUT, rnn_dropout=H.RNN_DROPOUT,
                        sample_rate=H.AUDIO_SAMPLE_RATE, window_size=H.SPECT_WINDOW_SIZE, initialize=torch_weight_init)
    if H.USE_CUDA:
        model.cuda()

    logging.info(model_summary(model, line_length=100))

    if H.PRELOAD_MODEL_PATH:
        state = torch.load(os.path.join(H.EXPERIMENT, H.PRELOAD_MODEL_PATH))
        model.load_state_dict(state)

    criterion = PytorchCTCLoss(cuda=False)  # WarpCTCLoss() #PytorchCTCLoss()

    # optimizer = optim.Adam(list(filter(lambda p:p.requires_grad, model.parameters())),
    #                         amsgrad = False,
    #                         betas = (0.9, 0.999),
    #                         eps = 1e-08,
    #                         lr = H.LR,
    #                         weight_decay = H.WEIGHT_DECAY)

    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())),
                          lr=H.LR, weight_decay=H.WEIGHT_DECAY, momentum=H.MOMENTUM, nesterov=H.NESTEROV)

    stopping = Stopping(model, patience=H.STOPPING_PATIENCE)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[H.LR_LAMBDA])

    ctc_decoder = CTCGreedyDecoder(vocab)

    scorer = Scorer(reduction='sum')

    tlogger = TensorboardLogger(root_dir=H.EXPERIMENT, experiment_dir=H.TIMESTAMP)  # PytorchLogger()

    checkpoint = Checkpoint(model, optimizer, stopping, m,
                            root_dir=H.EXPERIMENT, experiment_dir=H.TIMESTAMP, restore_from=-1,
                            interval=H.CHECKPOINT_INTERVAL, verbose=0)

    trainer = Trainer(model, train_loader, optimizer, scheduler, criterion, ctc_decoder, scorer, H.MAX_GRAD_NORM)

    evaluator = Evaluator(model, valid_loader, criterion, ctc_decoder, scorer)

    epoch_start = 1
    if H.CHECKPOINT_RESTORE:
        epoch_start = checkpoint.restore() + 1
        train_loader.batch_sampler.shuffle(epoch_start)

    epoch = epoch_start
    try:
        epoch_itr = tlogger.set_itr(range(epoch_start, H.MAX_EPOCHS + 1))

        for epoch in epoch_itr:

            with DelayedKeyboardInterrupt():

                m.train_loss, m.train_score, m.train_lr = trainer(epoch)

                m.valid_loss, m.valid_score = evaluator()

                if checkpoint:
                    checkpoint.step(epoch)

                stopping_flag = stopping.step(epoch, m.valid_loss, m.valid_score)

                epoch_itr.log_values(m.train_loss, m.train_score, m.train_lr,
                                     m.valid_loss, m.valid_score,
                                     stopping.best_score_epoch, stopping.best_score)

                if stopping_flag:
                    logger.info("Early stopping at epoch: %d, score %f" % (stopping.best_score_epoch,
                                                                           stopping.best_score))
                    break

                train_loader.batch_sampler.shuffle(epoch)

    except KeyboardInterrupt:
        logger.info("Training interrupted at: {}".format(epoch))
        pass

    checkpoint.create(epoch)

    model.load_state_dict(stopping.best_score_state)
    torch.save(model.state_dict(), os.path.join(H.EXPERIMENT, H.MODEL_NAME + '.tar'))

    logger.info(repr(tlogger))
    logger.info(repr(stopping))
    logger.info(repr(checkpoint))

    logger.info("Training end.")


#######################################################################################################################

def run_evaluation(H):

    create_logger(H)

    logger.info("Evaluation start.")

    random.seed(H.SEED)
    np.random.seed(H.SEED)
    torch.manual_seed(H.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(H.SEED)
        torch.cuda.manual_seed_all(H.SEED)

    vocab = Vocabulary(os.path.join(H.ROOT_DIR, H.EXPERIMENT), encoding=H.TARGET_ENCODING)

    audio_transform = transforms.Compose([
        AudioNormalizeDB(db=H.NORMALIZE_DB,
                         max_gain_db=H.NORMALIZE_MAX_GAIN),
        AudioSpectrogram(sample_rate=H.AUDIO_SAMPLE_RATE,
                         window_size=H.SPECT_WINDOW_SIZE,
                         window_stride=H.SPECT_WINDOW_STRIDE,
                         window=H.SPECT_WINDOW),
        AudioNormalize(),
        FromNumpyToTensor(tensor_type=torch.FloatTensor)
    ])

    label_transform = transforms.Compose([
        TranscriptEncodeCTC(vocab),
        FromNumpyToTensor(tensor_type=torch.LongTensor)
    ])

    test_dataset = AudioDataset(os.path.join(H.ROOT_DIR, H.EXPERIMENT), manifests_files=H.MANIFESTS, datasets="test",
                                transform=audio_transform, label_transform=label_transform, max_data_size=None,
                                sorted_by='recording_duration')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=H.BATCH_SIZE, num_workers=H.NUM_WORKERS,
                                              shuffle=False, collate_fn=collate_fn, pin_memory=True)

    logger.info(test_loader.dataset)

    model_pred = DeepSpeech2(len(vocab), rnn_hidden_size=H.RNN_HIDDEN_SIZE, nb_layers=H.NUM_LAYERS,
                             bidirectional=H.BIDIRECTIONAL, cnn_dropout=H.CNN_DROPOUT, rnn_dropout=H.RNN_DROPOUT,
                             sample_rate=H.AUDIO_SAMPLE_RATE, window_size=H.SPECT_WINDOW_SIZE,
                             initialize=torch_weight_init)
    if H.USE_CUDA:
        model_pred.cuda()

    state = torch.load(os.path.join(H.EXPERIMENT, H.MODEL_NAME + '.tar'))
    model_pred.load_state_dict(state)

    ctc_decoder = CTCGreedyDecoder(vocab)

    recognizer = Recognizer(model_pred, ctc_decoder, test_loader)

    hypotheses = recognizer()

    transcripts = []
    for _, labels, _, label_sizes, _ in test_loader:
        label_seq = CTCGreedyDecoder.decode_labels(labels, label_sizes, vocab)
        transcripts.extend(label_seq)

    bleu = Scorer.get_moses_multi_bleu(hypotheses, transcripts, lowercase=False)
    wer, cer = Scorer.get_wer_cer(hypotheses, transcripts)
    acc = Scorer.get_acc(hypotheses, transcripts)

    logger.info('Test Summary \n'
                'Bleu: {bleu:.3f}\n'
                'WER:  {wer:.3f}\n'
                'CER:  {cer:.3f}\n'
                'ACC:  {acc:.3f}'.format(bleu=bleu, wer=wer * 100, cer=cer * 100, acc=acc * 100))

#######################################################################################################################
