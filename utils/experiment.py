import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
import wandb

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel
from accelerate import Accelerator

from models.align import ModalityAlignment
from models.embed import DataEmbedding_ITS_Pooled
from models.llm import PEFTTSLLM
from models.tta import PEFTAdaINPatcher
from utils.norm import TSScaler
from metric import MetricManager

def train(args, trn_loader, val_loader, save_dir):

    # accelerator setting
    if args.cuda:
        accelerator = Accelerator(mixed_precision='bf16', log_with='wandb')
    else:
        accelerator = Accelerator(mixed_precision='bf16', cpu=True, log_with='wandb') # Adaptation, Evaluation mode

    accelerator.init_trackers(project_name="EHRTTA", config=vars(args), 
                              init_kwargs={"wandb": {"name": f"{args.model_id}_{args.seed}_{args.task_label}_{args.data_source}", "tags": ["dora"]}})
    global_step = 0

    device = accelerator.device

    # Define architectures
    model = PEFTTSLLM(args, device).to(device)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

    ts_embedder = DataEmbedding_ITS_Pooled(d_model=model.hidden_size, n_var = args.n_time_cols, device=device, 
                                           dropout=args.te_dropout, use_time=True).to(device)

    aligner = ModalityAlignment(d_model=model.hidden_size, n_heads=args.align_n_heads, 
                                dropout=args.align_dropout, use_gating=args.use_align_gate).to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    model, aligner, optimizer, trn_loader, val_loader  = accelerator.prepare(model, aligner, optimizer, trn_loader, val_loader)

    # initialize the normalization by using source training set
    scaler = TSScaler(args)
    scaler.reset_source_accum()
    
    for i, batch in enumerate(tqdm(trn_loader, desc='Initialize for the normalization')):
        _, x, mask, _, _, _, _ = batch
        x, mask = x.to(device), mask.to(device)
        scaler.update_source(x, mask)


    source_state = scaler.finalize_source()
    
    evaluator = MetricManager(args)

    print("Source mean/std ready", source_state["mean"].shape, source_state["std"].shape)

    # Training
    print('Training Start..')
    
    for epoch in range(1, args.n_epochs+1):
        print(f'==========[{epoch} / {args.n_epochs}]==========')
        
        running_trn_loss = 0.0
        running_val_loss = 0.0
        best_val_loss = 1e9

        evaluator.reset()
        model.train()

        for i, batch in enumerate(tqdm(trn_loader, desc='Source Training', total=len(trn_loader))):
            
            tt, x, ts_mask, input_ids, text_mask, y, pids = batch
            tt, x, ts_mask, input_ids, text_mask, y = tt.to(device), x.to(device), ts_mask.to(device), input_ids.to(device), text_mask.to(device), y.to(device)

            # 1) Time series Embedding 
            scaled_x = scaler.transform_source(x, ts_mask)    # 1-1) Time series Scaling
            ts_embedding = ts_embedder(tt, scaled_x, ts_mask) # 1-2) Time series Embedding -> (B,D,d_model)
            
            # 2) Convert text token id to text embedding vectors    
            text_embedding = model.backbone.get_input_embeddings()(input_ids) # (B, longest token length, d_model)

            # 3) Cross modality aligning
            aligned_embedding, cross_attn_weights = aligner(ts_embedding, text_embedding, text_mask, args.align_return_weights) # (B, D, d_model)

            # 4) embedding input
            with accelerator.accumulate(model):
                outputs = model(inputs_embeds=aligned_embedding, labels=y) # Dict(loss, logits, pooled : [(B, d_model)])
                
                loss = outputs['loss']

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                running_trn_loss  += loss.item()
                accelerator.log({"Train/loss": loss.item()}, step=global_step)
                global_step += 1

            logits, gt = accelerator.gather_for_metrics((outputs['logits'], y))

            if args.task == 'classification':
                evaluator.update_classification(logits, gt)
            else:
                evaluator.update_regression(logits, gt)
        
            if i % args.print_iter == 0 and i != 0:
                accelerator.print(f'[Epoch : {epoch},  Iter : {(i + 1):.3f} / {len(trn_loader):.3f}] Running Loss {running_trn_loss / (i + 1):.3f}')

        # Evaluation 
        train_loss = running_trn_loss / len(trn_loader)
        train_metrics = evaluator.compute()

        evaluator.reset() # reset for valiadation

        with torch.no_grad():
            model.eval()

            for i, batch in enumerate(tqdm(val_loader, desc='Source Validation', total=len(val_loader))):
                
                tt, x, ts_mask, input_ids, text_mask, y, pids = batch
                tt, x, ts_mask, input_ids, text_mask, y = tt.to(device), x.to(device), ts_mask.to(device), input_ids.to(device), text_mask.to(device), y.to(device)

                # 1) Time series Embedding 
                scaled_x = scaler.transform_source(x, ts_mask)    # 1-1) Time series Scaling
                ts_embedding = ts_embedder(tt, scaled_x, ts_mask) # 1-2) Time series Embedding -> (B,D,d_model)
                
                # 2) Tokenizing text data  
                text_embedding = model.get_input_embeddings()(input_ids) # (B, longest token length, d_model)

                # 3) Cross modality aligning
                aligned_embedding, cross_attn_weights = aligner(ts_embedding, text_embedding, text_mask, args.align_return_weights) # (B, D, d_model)

                # 4) embedding input
                with accelerator.accumulate(model):
                    outputs = model(inputs_embeds=aligned_embedding, labels=y) # Dict(loss, logits, pooled : [(B, d_model)])
                    
                    running_val_loss  += outputs['loss'].item()
                    
                    logits, gt = accelerator.gather_for_metrics((outputs['logits'], y))
                    
                    if args.task == 'classification':
                        evaluator.update_classification(logits, gt)
                    else:
                        evaluator.update_regression(logits, gt)

            valid_loss = running_val_loss / len(val_loader)
            valid_metrics = evaluator.compute()
        
        # Logging 
        print(f"====[Epoch {epoch+1}/{args.epochs} ] | Avg. Train Loss: {train_loss:.3f} | Avg. Valid Loss: {valid_loss:.3f}====")
        
        print(f"== Training Results == ")
        accelerator.print({'epoch' : epoch, **train_metrics})

        print(f"== Validation Results == ")
        accelerator.print({'epoch' : epoch, **valid_metrics})

        if args.task == 'classification':
            accelerator.log(
                {
                    "Epoch": epoch,
                    "Train/epoch_loss": train_loss,
                    "Valid/epoch_loss": valid_loss,
                    "Valid/auroc": valid_metrics["auroc"],
                    "Valid/auprc": valid_metrics["auprc"],
                    "Valid/f1" : valid_metrics['f1'],
                    "Valid/precision" : valid_metrics['precision'],
                    "Valid/recall" : valid_metrics['recall'],
                    "Valid/accuracy" : valid_metrics["accuracy"],
                    "Valid/confusion_metrix" : valid_metrics["confusion_matrix"],
                    "Train/auroc": train_metrics["auroc"],
                    "Train/auprc": train_metrics["auprc"],
                    "Train/f1" : train_metrics['f1'],
                    "Train/precision" : train_metrics['precision'],
                    "Train/recall" : train_metrics['recall'],
                    "Train/accuracy" : train_metrics["accuracy"],
                    "Train/confusion_metrix" : train_metrics["confusion_matrix"]
                }, step=global_step)
        else:
            accelerator.log(
                {
                    "Epoch": epoch,
                    "Train/epoch_loss": train_loss,
                    "Valid/epoch_loss": valid_loss,
                    "Valid/mse": valid_metrics["mse"],
                    "Valid/mae": valid_metrics["mae"],
                    "Valid/mape" : valid_metrics["mape"],
                    "Train/mse": train_metrics["mse"],
                    "Train/mae": train_metrics["mae"],
                    "Train/mape" : train_metrics["mape"]
                }, step=global_step)

        if best_val_loss > valid_loss:
            print(f'====[Epoch {epoch+1}/{args.epochs}] | Best Valid Loss update {best_val_loss:.3f} ==> {valid_loss:3f} ====')
            model.backbone.save_pretrained(save_dir)

    accelerator.end_training()
    print(f'==========[Finish]==========')

def inference(args, data_loader, save_dir):

    # Best model selection 

    return

def adaptation():

    return