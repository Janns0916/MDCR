import argparse
import math
import os
import sys
import time
from models.CCIM_Model import CCIM
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_lmkg import DBpedia
from dataset_rec import CRSRecDataset, CRSRecDataCollator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt
import pickle as pkl
import torch.nn as nn
# diffusion package
from src.diffusion_model.Use_Demoise import denoise_out

# 双gpu跑代码
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save/model/rec', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, default='redial_rec', help="A file containing all data.")
    parser.add_argument("--kg", type=str, default="lmkg", help="A file containing all knowledge graph data.")
    parser.add_argument("--kg_path", type=str, default="data/lmkg/lmkg.json", help="A file containing all knowledge graph data.")
    parser.add_argument("--image_path", type=str, default="data/lmkg/movientityId2imagemb.pkl", help="A file containing image enmbedding.")
    parser.add_argument("--use_image_embeds", default=True)  # action="store_true"
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--context_max_length", type=int, default=200, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int, default=200)
    parser.add_argument("--entity_max_length", type=int, default=64, help="max entity length in dataset.")
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument("--use_resp", default=True)
    parser.add_argument("--tokenizer", type=str, default="../../huggingface/DialoGPT-small")
    parser.add_argument("--text_tokenizer", type=str, default="../../huggingface/roberta-base")
    # model-
    parser.add_argument("--model", type=str, default="../../huggingface/DialoGPT-small",
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str, default="../../huggingface/roberta-base")
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    parser.add_argument("--n_prefix_rec", type=int, default=10)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--prompt_encoder", type=str, default="save/model/pre_trained_prompt/final")
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=4, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=64,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', default=530, type=int)
    parser.add_argument('--fp16', action='store_true')
    # wandb
    parser.add_argument("--use_wandb", type=bool, default=False, help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

    # diffusion model
    parser.add_argument('--w_min', type=float, default=0.1, help='the minimum weight for interactions')
    parser.add_argument('--w_max', type=float, default=1.0, help='the maximum weight for interactions')
    parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
    parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
    parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
    parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
    parser.add_argument('--entity_num', type=int, default=6423, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=True, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator(device_placement=False, fp16=args.fp16)
    device = accelerator.device

    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    # wandb
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)

        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()
    kg = DBpedia(kg=args.kg, debug=args.debug).get_entity_kg_info()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    # # 双gpu跑代码
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
        # model = torch.nn.DataParallel(model.cuda())
    model.to(device)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    # ***********************************************************************
    diffusion, denoise = denoise_out(args, device)
    # ***********************************************************************

    prompt_encoder = KGPrompt(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        n_prefix_rec=args.n_prefix_rec
    )

    if args.prompt_encoder is not None:
        prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)

    fix_modules = [model, text_encoder]
    for module in fix_modules:
        module.requires_grad_(False)

    # 让DialoGPT里的MHA和Linear进行优化
    for name, p in model.named_parameters():
        if name.startswith('mha') or name.startswith('image_proj'):
            p.requires_grad = True

    # optim & amp
    modules = [prompt_encoder]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # data
    train_dataset = CRSRecDataset(
        dataset=args.dataset, split='train', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
    )
    shot_len = int(len(train_dataset) * args.shot)
    train_dataset = random_split(train_dataset, [shot_len, len(train_dataset) - shot_len])[0]
    assert len(train_dataset) == shot_len
    valid_dataset = CRSRecDataset(
        dataset=args.dataset, split='valid', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
    )
    test_dataset = CRSRecDataset(
        dataset=args.dataset, split='test', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
    )
    data_collator = CRSRecDataCollator(
        tokenizer=tokenizer, device=device, debug=args.debug,
        context_max_length=args.context_max_length, entity_max_length=args.entity_max_length,
        pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    evaluator = RecEvaluator()
    prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0
    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # 读取相关实体
    kg_dir = os.path.join('data', args.kg)
    lmkg_moventityId2entitiesId = pkl.load(open(os.path.join(kg_dir, 'lmkg_moventityId2entitiesId.pkl'), 'rb'))
    lmkg_moventityId2imgemb = pkl.load(open(os.path.join(kg_dir, 'movientityId2imagemb.pkl'), 'rb'))
    # lmkg_moventityId2imgemb = pkl.load(open(os.path.join(kg_dir, 'lmkg_moventityId2imgemb.pkl'), 'rb'))

    # save model with best metric
    metric, mode = 'loss', -1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    # train loop
    for epoch in range(args.num_train_epochs):
        train_loss = []
        prompt_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                if args.use_image_embeds:
                    prompt_embeds = prompt_encoder(
                        entity_ids=batch['entity'],  # 64*12 里面都是entity_id
                        token_embeds=token_embeds,
                        output_entity=True,
                        use_rec_prefix=True
                    )
                    batch['context']['prompt_embeds'] = prompt_embeds
                    batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                    # 原始的推荐概率--这个是正常跑的
                    logits = model(**batch['context'], rec=True).rec_logits[:, kg['item_ids']]
                    # outputs = model(**batch['context'], rec=True)
                    # logits = outputs.rec_logits

                    prediction = diffusion.p_sample(denoise, logits, args.sampling_steps, args.sampling_noise)
                    ranks = torch.topk(prediction, k=50, dim=-1).indices.tolist()

                    # ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                    ranks = [[rank for rank in batch_rank] for batch_rank in ranks]

                    # 修改后的--这个还没试
                    # outputs = model(**batch['context'], rec=True)
                    # logits = outputs.rec_logits  # [:, selected_rows.long()]  64*6423
                    # ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                    # ranks = [[rank for rank in batch_rank] for batch_rank in ranks]

                    # 取出初次推荐电影的海报嵌入，若没有则为空
                    batch_imgemb = []
                    max_len = len(list(lmkg_moventityId2imgemb.values())[0])
                    for i, batch_one in enumerate(ranks):
                        pre_movie = batch_one[0]
                        if pre_movie in lmkg_moventityId2imgemb:
                            batch_imgemb.append(lmkg_moventityId2imgemb[pre_movie].detach().numpy())
                        else:
                            batch_imgemb.append(np.ones(max_len))

                    batch_imgemb = torch.tensor(np.array(batch_imgemb), dtype=torch.float32).cuda()  # 里面有64个子列表的ndarray格式，里面的每个维度是1024
                    # _, batch_imgemb = CCIM(args=args, prompt_embeds=prompt_embeds, batch_imagembs=batch_imgemb)

            prompt_embeds = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                image_embeds=batch_imgemb,
                output_entity=True,
                use_rec_prefix=True
            )

            batch['context']['prompt_embeds'] = prompt_embeds
            batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

            # # 修改内容
            items_ids = kg['item_ids']
            items_ids_tensor = torch.tensor(items_ids, dtype=torch.long, device='cuda')
            selected_rows = batch['context']['entity_embeds'][items_ids_tensor]
            batch['context']['movie_entity_embeds'] = torch.FloatTensor(selected_rows.cpu())

            # # 这里没修改的，正常跑的  扩散之前用的这个
            # outputs = model(**batch['context'], rec=True)
            # logits = outputs.rec_logits  # 64*6423
            # ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
            # # 这里跟zcy的不一样
            # # ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
            # ranks = [[rank for rank in batch_rank] for batch_rank in ranks]
            # loss = outputs.rec_loss / args.gradient_accumulation_steps
            # accelerator.backward(loss)
            # train_loss.append(float(loss))

            # 修改为扩散的内容
            outputs = model(**batch['context'], rec=True)
            logits = outputs.rec_logits  # 64*6423

            # diffusion_loss
            # *************************************************
            train_dl = diffusion.training_losses(denoise, logits, args.reweight)
            train_dl_loss = train_dl["loss"].mean()
            prediction = diffusion.p_sample(denoise, logits, args.sampling_steps, args.sampling_noise)
            ranks = torch.topk(prediction, k=50, dim=-1).indices.tolist()
            # *************************************************

            # ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
            # 这里跟zcy的不一样
            # ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
            ranks = [[rank for rank in batch_rank] for batch_rank in ranks]
            loss = outputs.rec_loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            train_loss.append(float(loss))

            # # 原始内容
            # loss = model(**batch['context'], rec=True).rec_loss / args.gradient_accumulation_steps
            # accelerator.backward(loss)
            # train_loss.append(float(loss))

            # optim step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        # metric
        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        # logger.info(f'epoch {epoch} train loss {train_loss}')
        logger.info(f'epoch {epoch} train loss {train_loss} diffusion loss {train_dl_loss}')

        del train_loss, batch

        # valid
        valid_loss = []
        prompt_encoder.eval()
        for batch in tqdm(valid_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                if args.use_image_embeds:
                    prompt_embeds = prompt_encoder(
                        entity_ids=batch['entity'],  # 64*12 里面都是entity_id
                        token_embeds=token_embeds,
                        output_entity=True,
                        use_rec_prefix=True
                    )
                    batch['context']['prompt_embeds'] = prompt_embeds
                    batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                    # 这里是用的下面的两个，不正常的话，用上面那一句
                    logits = model(**batch['context'], rec=True).rec_logits[:, kg['item_ids']]
                    # logits = model(**batch['context'], rec=True).rec_logits

                    prediction = diffusion.p_sample(denoise, logits, args.sampling_steps, args.sampling_noise)
                    ranks = torch.topk(prediction, k=50, dim=-1).indices.tolist()

                    # ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                    ranks = [[rank for rank in batch_rank] for batch_rank in ranks]

                    # 取出初次推荐电影的海报嵌入，若没有则为空
                    batch_imgemb = []
                    max_len = len(list(lmkg_moventityId2imgemb.values())[0])
                    for i, batch_one in enumerate(ranks):
                        pre_movie = batch_one[0]
                        if pre_movie in lmkg_moventityId2imgemb:
                            batch_imgemb.append(lmkg_moventityId2imgemb[pre_movie].detach().numpy())
                        else:
                            batch_imgemb.append(np.ones(max_len))
                    batch_imgemb = torch.tensor(np.array(batch_imgemb), dtype=torch.float32).cuda()
                    # _, batch_imgemb = CCIM(args=args, prompt_embeds=prompt_embeds, batch_imagembs=batch_imgemb)

                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    image_embeds=batch_imgemb,
                    output_entity=True,
                    use_rec_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                # 修改内容
                outputs = model(**batch['context'], rec=True)
                valid_loss.append(float(outputs.rec_loss))

                logits = outputs.rec_logits[:, kg['item_ids']]

                # diffusion_loss
                # *************************************************
                valid_dl = diffusion.training_losses(denoise, logits, args.reweight)
                valid_dl_loss = valid_dl["loss"].mean()
                prediction = diffusion.p_sample(denoise, logits, args.sampling_steps, args.sampling_noise)
                ranks = torch.topk(prediction, k=50, dim=-1).indices.tolist()
                # *************************************************

                # ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[rank for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['movie_rec_labels']
                # labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

                # 原始内容
                # outputs = model(**batch['context'], rec=True)
                # valid_loss.append(float(outputs.rec_loss))
                # logits = outputs.rec_logits[:, kg['item_ids']]
                # ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                # ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                # labels = batch['context']['rec_labels']
                # evaluator.evaluate(ranks, labels)

        # metric
        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        valid_report = {}
        for k, v in report.items():
            if k != 'count':
                valid_report[f'valid/{k}'] = v / report['count']
        valid_report['valid/loss'] = np.mean(valid_loss)
        # valid_report['valid_dl_loss'] = valid_dl_loss
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            prompt_encoder.save(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')

        # test
        test_loss = []
        prompt_encoder.eval()
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                if args.use_image_embeds:

                    prompt_embeds = prompt_encoder(
                        entity_ids=batch['entity'],  # 64*12 里面都是entity_id
                        token_embeds=token_embeds,
                        output_entity=True,
                        use_rec_prefix=True
                    )
                    batch['context']['prompt_embeds'] = prompt_embeds
                    batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                    logits = model(**batch['context'], rec=True).rec_logits[:, kg['item_ids']]
                    # logits = model(**batch['context'], rec=True).rec_logits

                    prediction = diffusion.p_sample(denoise, logits, args.sampling_steps, args.sampling_noise)
                    ranks = torch.topk(prediction, k=50, dim=-1).indices.tolist()

                    # ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                    ranks = [[rank for rank in batch_rank] for batch_rank in ranks]

                    batch_imgemb = []
                    max_len = len(list(lmkg_moventityId2imgemb.values())[0])
                    for i, batch_one in enumerate(ranks):
                        pre_movie = batch_one[0]
                        if pre_movie in lmkg_moventityId2imgemb:
                            batch_imgemb.append(lmkg_moventityId2imgemb[pre_movie].detach().numpy())
                        else:
                            batch_imgemb.append(np.ones(max_len))
                    batch_imgemb = torch.tensor(np.array(batch_imgemb), dtype=torch.float32).cuda()
                    # _, batch_imgemb = CCIM(args=args, prompt_embeds=prompt_embeds, batch_imagembs=batch_imgemb)

                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    image_embeds=batch_imgemb,
                    output_entity=True,
                    use_rec_prefix=True
                )

                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                # # 修改内容
                outputs = model(**batch['context'], rec=True)
                test_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits[:, kg['item_ids']]

                # diffusion_loss
                # *************************************************
                test_dl = diffusion.training_losses(denoise, logits, args.reweight)
                test_dl_loss = test_dl["loss"].mean()
                prediction = diffusion.p_sample(denoise, logits, args.sampling_steps, args.sampling_noise)
                ranks = torch.topk(prediction, k=50, dim=-1).indices.tolist()
                # *************************************************

                # logits = outputs.rec_logits
                # ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[rank for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['movie_rec_labels']

                evaluator.evaluate(ranks, labels)

                # 原始内容
                # outputs = model(**batch['context'], rec=True)
                # test_loss.append(float(outputs.rec_loss))
                # logits = outputs.rec_logits[:, kg['item_ids']]
                # ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                # ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                # labels = batch['context']['rec_labels']
                # evaluator.evaluate(ranks, labels)
        # metric
        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        test_report = {}
        for k, v in report.items():
            if k != 'count':
                test_report[f'test/{k}'] = v / report['count']
        test_report['test/loss'] = np.mean(test_loss)
        # test_report['test_dl_loss'] = test_dl_loss
        test_report['epoch'] = epoch
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()

    final_dir = os.path.join(args.output_dir, 'final')
    prompt_encoder.save(final_dir)
    logger.info(f'save final model')
