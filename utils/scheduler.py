import numpy as np
import math


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule



    # #    parser.add_argument('--weight_decay', type=float, default=0.05,
    #                     help='weight decay (default: 0.05)')
    # parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
    #     weight decay. We use a cosine schedule for WD and using a larger decay by
    #     the end of training improves performance for ViTs.""")
    # parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
    #                     help='learning rate (default: 4e-3), with total batch size 4096')
    # parser.add_argument('--layer_decay', type=float, default=1.0)
    # parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    # parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
    #                     help='epochs to warmup LR, if scheduler supports')
    # parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
    #                     help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    #
    # print("Use Cosine LR scheduler")
    # lr_schedule_values = utils.cosine_scheduler(
    #     args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
    #     warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    # )
    #
    # if args.weight_decay_end is None:
    #     args.weight_decay_end = args.weight_decay
    # wd_schedule_values = utils.cosine_scheduler(
    #     args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    # print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
#
# num_training_steps_per_epoch = len(dataset_train) // total_batch_size
# if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
#     for i, param_group in enumerate(optimizer.param_groups):
#         if lr_schedule_values is not None:
#             param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
#         if wd_schedule_values is not None and param_group["weight_decay"] > 0:
#             param_group["weight_decay"] = wd_schedule_values[it]
