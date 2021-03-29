import torch
import logging
import os
from models import build_model, Checkpointer, PeriodicCheckpointer
from solver import build_lr_scheduler, build_optimizer
from data import build_reid_test_loader, build_reid_train_loader
from evaluation import ReidEvaluator, inference_on_dataset
from configs import cfg
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate

def main(cfg):
    model = build_model(cfg)
    model.cuda()

    log_file = os.path.join(log_dir, '{}.txt'.format(cfg.log_name))
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(cfg)

    if cfg.eval_only:
        cfg.pretrained = False
        Checkpointer(model).load(cfg.resume_weights)
        test(cfg, model, logger)
    train(cfg, model, logger)
    test(cfg, model, logger)


def train(cfg, model, logger):
    logger.info('Start Training....')
    train_data_loader = build_reid_train_loader(cfg)
    test_data_loader = build_reid_test_loader(cfg)
    optimizer = build_optimizer(cfg, model)
    optimizer_ckpt = dict(optimizer=optimizer)
    iters_per_epoch = len(train_data_loader.dataset) // (cfg.batch_id * cfg.batch_image)
    scheduler = build_lr_scheduler(cfg, optimizer, iters_per_epoch)

    max_epoch = cfg.max_epoch
    max_iter = max_epoch * iters_per_epoch
    warmup_iters = cfg.warmup_iters
    delay_epoch = cfg.delay_epochs
    log_dir = cfg.log_dir
    os.mkdir(log_dir)
    save_dir = cfg.save_dir
    os.mkdir(save_dir)
    writer = SummaryWriter(log_dir)

    checkpointer = Checkpointer(
            model,
            save_dir,
            **optimizer_ckpt,
            **scheduler
        )

    start_epoch = (
                checkpointer.resume_or_load(cfg.resume_weights, resume=cfg.resume).get("epoch", -1) + 1
        )
    iteration = start_iter = start_epoch * iters_per_epoch

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.save_interval, max_epoch)

    logger.info("Start training from epoch {}".format(start_epoch))
    for epoch in range(start_epoch, max_epoch):
        model.train()
        for data, _ in zip(train_data_loader, range(iters_per_epoch)):
            accuracy, loss_dict = model(data)
            for k, v in loss_dict.items():
                writer.add_scalar(tag=k, value=(k, v), global_step=iteration)
            writer.add_scalar(tag='accuracy', value=('accuracy', accuracy), global_step=iteration)
            losses = sum(loss_dict.values())
            if iteration % cfg.log_interval == 0: 
                logger.info('Epoch: [%d], Iteration: [%d], LR : %.3f, Loss: %.4f, Accuracy: %.4f'
                        %(epoch, iteration, optimizer.param_groups[0]["lr"], losses, accuracy))

            optimizer.zero_grad()

            losses.backward()

            optimizer.step()

            iteration += 1

            if iteration <= warmup_iters:
                scheduler["warmup_sched"].step()

        if iteration > warmup_iters and (epoch + 1) >= delay_epochs:
            scheduler["lr_sched"].step()

        if (epoch+1) % cfg.eval_interval == 0:
            test(cfg, model, logger)
        periodic_checkpointer.step(epoch) 

        

def get_evaluator(cfg, dataset_name, output_dir=None):
    data_loader, num_query = build_reid_test_loader(cfg, dataset_name)
    return data_loader, ReidEvaluator(cfg, num_query, dataset=None, output_dir=output_dir)

def test(cfg, model, logger):
    logger.info('Start Evaluating....')
    results = OrderedDict()
    for idx, dataset_name in enumerate(cfg.datasets_test):
        data_loader, evaluator = get_evaluator(cfg, dataset_name)
        results_i = inference_on_dataset(model, data_loader, evaluator, dataset=None, flip_test=cfg.flip_test)
        results[dataset_name] = results_i
    print_csv_format(results, logger)
    if len(results) == 1: results = list(results.values())[0]

    return results

def print_csv_format(results, logger):
    assert isinstance(results, OrderedDict), results
    task = list(results.keys())[0]
    metrics = ["Datasets"] + [k for k in results[task]]

    csv_results = []
    for task, res in results.items():
        csv_results.append((task, *list(res.values())))

    # tabulate it
    table = tabulate(
        csv_results,
        tablefmt="pipe",
        floatfmt=".4f",
        headers=metrics,
        numalign="left",
    )

    logger.info("Evaluation results in csv format: \n" + colored(table, "cyan"))

if __name__ == "__main__":
    main(cfg)
