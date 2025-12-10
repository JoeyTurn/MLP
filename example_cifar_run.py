import numpy as np
import torch

import torch.multiprocessing as mp
from backend.cli import parse_args, build_other_grabs

from data.monomial import Monomial
from data.data import polynomial_batch_fn
from backend.job_iterator import main as run_job_iterator
from backend.utils import ensure_torch, load_json

from .ntk_coeffs import get_relu_level_coeff_fn

import os, sys
from FileManager import FileManager


if __name__ == "__main__":

    args = parse_args() #default args

    # Set any args that we want to differ
    args.ONLINE = True
    args.N_TRAIN=4000
    args.N_TEST=1000
    args.NUM_TRIALS = 3
    args.N_TOT = args.N_TEST+args.N_TRAIN
    
    datapath = os.getenv("DATASETPATH") #datapath = os.path.join(os.getenv(...))
    exptpath = os.getenv("EXPTPATH") #same here
    if datapath is None:
        raise ValueError("must set $DATASETPATH environment variable")
    if exptpath is None:
        raise ValueError("must set $EXPTPATH environment variable")
    expt_name = "example_mlp_run"
    dataset = "synthetic"
    expt_dir = os.path.join(exptpath, "example_folder", expt_name, dataset)

    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
    expt_fm = FileManager(expt_dir)
    print(f"Working in directory {expt_dir}.")

    
    if args.TARGET_MONOMIALS is not None:
        args.TARGET_MONOMIALS = [Monomial(m) for m in args.TARGET_MONOMIALS]
    elif args.target_monomials_path:
        target_monomials_json = load_json(args.target_monomials_path)
        args.TARGET_MONOMIALS = [Monomial(m) for m in target_monomials_json]
    elif args.TARGET_MONOMIALS is None:
        args.TARGET_MONOMIALS = [Monomial({10: 1}), Monomial({190:1}), Monomial({0:2}), Monomial({2:1, 3:1}), Monomial({15:1, 20:1}), Monomial({0:3}),]
        
    if args.datasethps_path:
        args.datasethps = load_json(args.datasethps_path)

    from ImageData import ImageData
    PIXEL_NORMALIZED =  False # Don't normalize pixels, normalize samples
    classes = args.datasethps['classes']
    normalized = args.datasethps['normalized']

    if classes is not None:
        imdata = ImageData('cifar10', "../data", classes=classes, onehot=len(classes)!=2, format="N")
    else:
        imdata = ImageData('cifar10', "../data", classes=classes, onehot=False, format="N")
    X_train, y_train = imdata.get_dataset(args.N_TRAIN, **args.datasethps, get='train',
                                        centered=True, normalize=PIXEL_NORMALIZED)
    X_test, y_test = imdata.get_dataset(args.N_TEST, **args.datasethps, get='test',
                                        centered=True, normalize=PIXEL_NORMALIZED)
    X_train, y_train, X_test, y_test = map(ensure_torch, (X_train, y_train, X_test, y_test))
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    X_train, y_train, X_test, y_test = [t/torch.linalg.norm(t) for t in (X_train, y_train, X_test, y_test)] if normalized else (X_train, y_train, X_test, y_test)
    if normalized:
        X_train *= args.N_TRAIN**(0.5); X_test *= args.N_TEST**(0.5)
        y_train *= args.N_TRAIN**(0.5); y_test *= args.N_TEST**(0.5)
    X_full = torch.cat((X_train, X_test), dim=0)
    y_full = torch.cat((y_train, y_test), dim=0)
    data_eigvals = torch.linalg.svdvals(X_full)**2
    data_eigvals /= data_eigvals.sum()

    U, lambdas, Vt = torch.linalg.svd(X_full, full_matrices=False)
    dim = X_full.shape[1]

    ## --- Target function defs ---
    if args.TARGET_FUNCTION_TYPE == "monomial":
        target_monomials = args.TARGET_MONOMIALS
        targets = target_monomials
        bfn_config = dict(lambdas=lambdas, Vt=Vt, data_eigvals=data_eigvals, N=args.N_TOT)
    

    batch_function = lambda n, X, y, gen, target_monomial: polynomial_batch_fn(**bfn_config, monomials=target_monomial, bsz=n, gen=gen, X=X, y=y)

    global_config = dict(DEPTH=args.DEPTH, WIDTH=args.WIDTH, LR=args.LR, GAMMA=args.GAMMA,
        EMA_SMOOTHER=args.EMA_SMOOTHER, MAX_ITER=args.MAX_ITER,
        LOSS_CHECKPOINTS=args.LOSS_CHECKPOINTS, N_TEST=args.N_TEST,
        SEED=args.SEED, ONLYTHRESHOLDS=args.ONLYTHRESHOLDS, DIM=dim,
        TARGET_FUNCTION_TYPE=args.TARGET_FUNCTION_TYPE,
        ONLINE=args.ONLINE, VERBOSE=args.VERBOSE
        )

    grabs = build_other_grabs(args.other_model_grabs, default_source=args.W_source, concat_outside=args.concat_outside,
        per_alias_gram=args.other_model_gram, per_alias_kwargs=args.other_model_kwargs,)
    global_config.update({"otherreturns": grabs})
    
    mp.set_start_method("spawn", force=True)
    sample_sizes = [100, 500, 1000]
    trials = [0, 1, 2]
    
    iterators = [args.N_SAMPLES, args.NUM_TRIALS, args.TARGET_MONOMIALS]
    iterator_names = ["ntrain", "trial", "target_monomial"]
    

    result = run_job_iterator(iterators, iterator_names, global_config, bfn=batch_function)
    expt_fm.save(result, "result.pickle")
    torch.cuda.empty_cache()