#coding=utf8
import sys, os, time, json, gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import Namespace
from utils.args import init_args
from utils.hyperparams import hyperparam_path
from utils.initialization import *
from utils.example import Example
from utils.batch import Batch
from utils.optimization import set_optimizer
from model.model_utils import Registrable
from model.model_constructor import *

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])

# TODO CL
# if args.exp_path:
#     exp_path = os.path.join('exp', args.exp_path)
#     if not os.path.exists(exp_path):
#         os.makedirs(exp_path)
# else:
#     exp_path = hyperparam_path(args)
exp_path = hyperparam_path(args)

logger = set_logger(exp_path, args.testing)
set_random_seed(args.seed)
device = set_torch_device(args.device)
logger.info("Initialization finished ...")
logger.info("Output path is %s" % (exp_path))
logger.info("Random seed is set to %d" % (args.seed))
logger.info("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

# load dataset and vocabulary
start_time = time.time()
if args.read_model_path:
    params = json.load(open(os.path.join(args.read_model_path, 'params.json')), object_hook=lambda d: Namespace(**d))
    params.lazy_load = True
else:
    params = args
# set up the grammar, transition system, evaluator, etc.

data_dir = params.data_dir  # TODO
Example.configuration(plm=params.plm, method=params.model, 
                      table_path=os.path.join(data_dir, 'tables.json'), 
                      tables=os.path.join(data_dir, 'tables.bin'), 
                      db_dir=os.path.join(data_dir, 'database'))
train_dataset, dev_dataset = Example.load_dataset('train', data_dir=data_dir), Example.load_dataset('dev', data_dir=data_dir)
# test_dataset = Example.load_dataset('test', data_dir=data_dir)  # TODO
logger.info("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
logger.info("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))
# logger.info("Dataset size: train -> %d ; dev -> %d ; test -> %d" % (len(train_dataset), len(dev_dataset), len(test_dataset)))   # TODO
sql_trans, evaluator = Example.trans, Example.evaluator
args.word_vocab, args.relation_num = len(Example.word_vocab), len(Example.relation_vocab)

# model init, set optimizer
model = Registrable.by_name('text2sql')(params, sql_trans).to(device)

# TODO count model params num
# def getModelSize(model):
#     param_size = 0
#     param_sum = 0
#     for param in model.parameters():
#         param_size += param.nelement() * param.element_size()
#         param_sum += param.nelement()
#     buffer_size = 0
#     buffer_sum = 0
#     for buffer in model.buffers():
#         buffer_size += buffer.nelement() * buffer.element_size()
#         buffer_sum += buffer.nelement()
#     all_size = (param_size + buffer_size) / 1024 / 1024
#     print('模型总大小为：{:.3f}MB'.format(all_size))
#     return (param_size, param_sum, buffer_size, buffer_sum, all_size)
# print(getModelSize(model))
# total = sum([param.nelement() for param in model.parameters()])
# print("Number of parameter: %.2fM" % (total/1e6))
# import pdb; pdb.set_trace()

if args.read_model_path:
    check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    logger.info("Load saved model from path: %s" % (args.read_model_path))
else:
    json.dump(vars(params), open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
    if params.plm is None:
        ratio = Example.word2vec.load_embeddings(model.encoder.input_layer.word_embed, Example.word_vocab, device=device)
        logger.info("Init model and word embedding layer with a coverage %.2f" % (ratio))
# logger.info(str(model))

def decode(choice, output_path, acc_type='sql', use_checker=False):
    assert acc_type in ['beam', 'ast', 'sql'] and choice in ['train', 'dev', 'test']
    model.eval()
    # dataset = train_dataset if choice == 'train' else dev_dataset
    # TODO
    if choice =='train':
        dataset = train_dataset
    elif choice == 'dev':
        dataset = dev_dataset
    # elif choice == 'test':
    #     dataset = test_dataset

    all_hyps = []
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            current_batch = Batch.from_example_list(dataset[i: i + args.batch_size], device, train=False)
            hyps = model.parse(current_batch, args.beam_size)
            all_hyps.extend(hyps)
        acc = evaluator.acc(all_hyps, dataset, output_path, acc_type=acc_type, etype='match', use_checker=use_checker)
    torch.cuda.empty_cache()
    gc.collect()
    return acc

# TODO CL
if args.CL_mode:
    CL_EPOCH_NUM = args.CL_epoch_num
    START_CL_EPOCH = int(args.warmup_ratio * args.max_epoch)

if args.CL_mode == 'soft':
    GAMMA0 = 0.1
    ALPHA0 = 0.5
    SOFT_CL_WEIGHT_LS = [None, ]

    for i in range(1, CL_EPOCH_NUM + 1):
        SOFT_CL_WEIGHT_LS.append([None, None, ])
        gamma_i = GAMMA0 + i / CL_EPOCH_NUM * (1 - GAMMA0)
        for l in range(2, 100):
            SOFT_CL_WEIGHT_LS[-1].append([])
            for t in range(1, l + 1):
                alpha_t_l = ALPHA0 * (t - 1) / (l - 1)
                weight = gamma_i ** (alpha_t_l * args.soft_CL_matrix_decay)
                SOFT_CL_WEIGHT_LS[-1][-1].append(weight)


if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    logger.info('Total training steps: %d;\t Warmup steps: %d' % (num_training_steps, num_warmup_steps))
    optimizer, scheduler = set_optimizer(model, args, num_warmup_steps, num_training_steps)
    # start_epoch, nsamples, best_result = 0, len(train_dataset), {'dev_acc': 0.0, 'test_acc': 0.0} 
    start_epoch, nsamples, best_result = 1, len(train_dataset), {'dev_acc': 0.0, 'test_acc': 0.0}   # TODO CL
    train_index, step_size = np.arange(nsamples), args.batch_size // args.grad_accumulate
    if args.read_model_path and args.load_optimizer:
        optimizer.load_state_dict(check_point['optim'])
        scheduler.load_state_dict(check_point['scheduler'])
        start_epoch = check_point['epoch'] + 1
    logger.info('Start training ......')
    for i in range(start_epoch, args.max_epoch + 1):

        # TODO CL
        if args.CL_mode == 'soft':
            if i - START_CL_EPOCH > 0 and i - START_CL_EPOCH < CL_EPOCH_NUM + 1:
                logger.info('is soft CL epoch...')

        start_time = time.time()
        epoch_loss, epoch_gp_loss, count = 0, 0, 0
        np.random.shuffle(train_index)
        model.train()
        for j in range(0, nsamples, step_size):
            count += 1
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = Batch.from_example_list(cur_dataset, device, train=True, smoothing=args.smoothing)
            
            # TODO CL
            if args.CL_mode == 'soft':
                if i - START_CL_EPOCH > 0 and i - START_CL_EPOCH < CL_EPOCH_NUM + 1:
                    # epoch 31 → 70
                    soft_cl_ls = SOFT_CL_WEIGHT_LS[i - START_CL_EPOCH]
                    loss, gp_loss = model(current_batch, soft_cl_ls=soft_cl_ls)
                else:
                    loss, gp_loss = model(current_batch)
            else:
                loss, gp_loss = model(current_batch) # see utils/batch.py for batch elements
            
            epoch_loss += loss.item()
            epoch_gp_loss += gp_loss.item()
            # print("Minibatch loss: %.4f" % (loss.item()))
            loss += gp_loss
            loss.backward()
            if count == args.grad_accumulate or j + step_size >= nsamples:
                count = 0
                model.pad_embedding_grad_zero()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        logger.info('Training: \tEpoch: %d\tTime: %.4f\tTraining loss: %.4f/%.4f' % (i, time.time() - start_time, epoch_loss, epoch_gp_loss))
        torch.cuda.empty_cache()
        gc.collect()

        if i < args.eval_after_epoch: # avoid unnecessary evaluation
            continue


        start_time = time.time()
        dev_acc = decode('dev', os.path.join(exp_path, 'dev.iter' + str(i)), acc_type='sql')
        logger.info('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.4f' % (i, time.time() - start_time, dev_acc))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_acc'], best_result['iter'] = dev_acc, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, open(os.path.join(exp_path, 'model.bin'), 'wb'))
            logger.info('NEW BEST MODEL: \tEpoch: %d\tDev acc: %.4f' % (i, dev_acc))

        # TODO
        # start_time = time.time()
        # dev_acc = decode('dev', os.path.join(exp_path, 'dev.iter' + str(i)), acc_type='sql')
        # logger.info('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.4f' % (i, time.time() - start_time, dev_acc))
        # start_time = time.time()
        # test_acc = decode('test', os.path.join(exp_path, 'test.iter' + str(i)), acc_type='sql')
        # logger.info('Evaluation: \tEpoch: %d\tTime: %.4f\tTest acc: %.4f' % (i, time.time() - start_time, test_acc))
        # if dev_acc > best_result['dev_acc']:
        #     best_result['dev_acc'], best_result['iter'] = dev_acc, i
        #     torch.save({
        #         'epoch': i, 'model': model.state_dict(),
        #         'optim': optimizer.state_dict(),
        #         'scheduler': scheduler.state_dict()
        #     }, open(os.path.join(exp_path, 'model.bin'), 'wb'))
        #     logger.info('NEW BEST Dev: \tEpoch: %d\tDev acc: %.4f' % (i, dev_acc))
        # if test_acc > best_result['test_acc']:
        #     best_result['test_acc'], best_result['iter'] = test_acc, i
        #     logger.info('NEW BEST Test: \tEpoch: %d\tTest acc: %.4f' % (i, test_acc))

    logger.info('FINAL BEST RESULT: \tEpoch: %d\tDev acc: %.4f' % (best_result['iter'], best_result['dev_acc']))
    # logger.info('FINAL BEST Dev: \tEpoch: %d\tDev acc: %.4f' % (best_result['iter'], best_result['dev_acc']))       # TODO
    # logger.info('FINAL BEST TEST: \tEpoch: %d\tTest acc: %.4f' % (best_result['iter'], best_result['test_acc']))    # TODO

    # check_point = torch.load(open(os.path.join(exp_path, 'model.bin'), 'rb'))
    # model.load_state_dict(check_point['model'])
    # dev_acc_beam = decode('dev', output_path=os.path.join(exp_path, 'dev.iter' + str(best_result['iter']) + '.beam' + str(args.beam_size)), acc_type='beam')
    # logger.info('FINAL BEST RESULT: \tEpoch: %d\tDev acc/Beam acc: %.4f/%.4f' % (best_result['iter'], best_result['dev_acc'], dev_acc_beam))
else:
    # start_time = time.time()
    # train_acc = decode('train', output_path=os.path.join(args.read_model_path, 'train.eval'), acc_type='sql')
    # logger.info("Evaluation costs %.2fs ; Train dataset exact match acc is %.4f ." % (time.time() - start_time, train_acc))
    start_time = time.time()
    dev_acc = decode('dev', output_path=os.path.join(args.read_model_path, 'dev.eval'), acc_type='sql')
    dev_acc_checker = decode('dev', output_path=os.path.join(args.read_model_path, 'dev.eval.checker'), acc_type='sql', use_checker=True)
    dev_acc_beam = decode('dev', output_path=os.path.join(args.read_model_path, 'dev.eval.beam' + str(args.beam_size)), acc_type='beam')
    logger.info("Evaluation costs %.2fs ; Dev dataset exact match/checker/beam acc is %.4f/%.4f/%.4f ." % (time.time() - start_time, dev_acc, dev_acc_checker, dev_acc_beam))
