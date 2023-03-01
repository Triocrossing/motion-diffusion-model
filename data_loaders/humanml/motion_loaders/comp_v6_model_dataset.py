import torch
from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
from utils import dist_util
import collections
import pickle
import os.path as osp


def load_pickle_motion(pickle_path):
    """Load a pickle file."""
    if not osp.exists(pickle_path):
        return None
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_motions(
    fname,
    motions,
    folder_dir="/media/xi/ssd02/Work/textual_inversion/benchmark/save_motion",
):
    import ipdb

    ipdb.set_trace()
    ctr = 0
    for idx, name in enumerate(fname):
        randidx = np.random.randint(10)
        _motion = load_pickle_motion(osp.join(folder_dir, f"{name}_{randidx}.pkl"))

        if _motion is not None:
            motions[idx] = _motion.clone()
            ctr += 1
    print(f"percentage of motion existed: {float(ctr)/len(fname)}")
    return motions


def load_motion(
    fname,
    motions,
    folder_dir="/media/xi/ssd02/Work/textual_inversion/benchmark/save_motion",
):

    randidx = np.random.randint(10)
    _motion = load_pickle_motion(osp.join(folder_dir, f"{fname}_{randidx}.pkl"))

    if _motion is not None:
        return _motion, True

    # print(f"percentage of motion existed: {float(ctr)/len(fname)}")
    return None, False


def load_pickle(pickle_path):
    """Load a pickle file."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def build_models(opt):
    if opt.text_enc_mod == "bigru":
        text_encoder = TextEncoderBiGRU(
            word_size=opt.dim_word,
            pos_size=opt.dim_pos_ohot,
            hidden_size=opt.dim_text_hidden,
            device=opt.device,
        )
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception("Text Encoder Mode not Recognized!!!")

    seq_prior = TextDecoder(
        text_size=text_size,
        input_size=opt.dim_att_vec + opt.dim_movement_latent,
        output_size=opt.dim_z,
        hidden_size=opt.dim_pri_hidden,
        n_layers=opt.n_layers_pri,
    )

    seq_decoder = TextVAEDecoder(
        text_size=text_size,
        input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
        output_size=opt.dim_movement_latent,
        hidden_size=opt.dim_dec_hidden,
        n_layers=opt.n_layers_dec,
    )

    att_layer = AttLayer(
        query_dim=opt.dim_pos_hidden, key_dim=text_size, value_dim=opt.dim_att_vec
    )

    movement_enc = MovementConvEncoder(
        opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent
    )
    movement_dec = MovementConvDecoder(
        opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose
    )

    len_estimator = MotionLenEstimatorBiGRU(
        opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes
    )

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)
    checkpoints = torch.load(
        pjoin(
            opt.checkpoints_dir,
            opt.dataset_name,
            "length_est_bigru",
            "model",
            "latest.tar",
        ),
        map_location=opt.device,
    )
    len_estimator.load_state_dict(checkpoints["estimator"])
    len_estimator.to(opt.device)
    len_estimator.eval()

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return (
        text_encoder,
        seq_prior,
        seq_decoder,
        att_layer,
        movement_enc,
        movement_dec,
        len_estimator,
    )


class CompV6GeneratedDataset(Dataset):
    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        (
            text_enc,
            seq_pri,
            seq_dec,
            att_layer,
            mov_enc,
            mov_dec,
            len_estimator,
        ) = build_models(opt)
        trainer = CompTrainerV6(
            opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc
        )
        epoch, it, sub_ep, schedule_len = trainer.load(
            pjoin(opt.model_dir, opt.which_epoch + ".tar")
        )
        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 10 if opt.dataset_name == "t2m" else 6
        # print(mm_idxs)

        print("Loading model: Epoch %03d Schedule_len %03d" % (epoch, schedule_len))
        trainer.eval_mode()
        trainer.to(opt.device)
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                tokens = tokens[0].split("_")
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
                pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

                mm_num_now = len(mm_generated_motions)
                is_mm = (
                    True
                    if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now]))
                    else False
                )

                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)

                    m_lens = mov_length * opt.unit_length
                    pred_motions, _, _ = trainer.generate(
                        word_emb,
                        pos_ohot,
                        cap_lens,
                        m_lens,
                        m_lens[0] // opt.unit_length,
                        opt.dim_pose,
                    )
                    if t == 0:
                        # print(m_lens)
                        # print(text_data)
                        sub_dict = {
                            "motion": pred_motions[0].cpu().numpy(),
                            "length": m_lens[0].item(),
                            "cap_len": cap_lens[0].item(),
                            "caption": caption[0],
                            "tokens": tokens,
                        }
                        generated_motion.append(sub_dict)

                    if is_mm:
                        mm_motions.append(
                            {
                                "motion": pred_motions[0].cpu().numpy(),
                                "length": m_lens[0].item(),
                            }
                        )
                if is_mm:
                    mm_generated_motions.append(
                        {
                            "caption": caption[0],
                            "tokens": tokens,
                            "cap_len": cap_lens[0].item(),
                            "mm_motions": mm_motions,
                        }
                    )

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.opt = opt
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = (
            data["motion"],
            data["length"],
            data["caption"],
            data["tokens"],
        )
        sent_len = data["cap_len"]

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.opt.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.opt.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )
        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
        )


class CompMDMGeneratedDataset(Dataset):
    def __init__(
        self,
        model,
        diffusion,
        dataloader,
        mm_num_samples,
        mm_num_repeats,
        max_motion_length,
        num_samples_limit,
        scale=1.0,
    ):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print("real_num_batches", real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(
                real_num_batches,
                mm_num_samples // dataloader.batch_size + 1,
                replace=False,
            )
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print("mm_idxs", mm_idxs)

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):
                if (
                    num_samples_limit is not None
                    and len(generated_motion) >= num_samples_limit
                ):
                    break

                tokens = [t.split("_") for t in model_kwargs["y"]["tokens"]]

                # add CFG scale to batch
                if scale != 1.0:
                    model_kwargs["y"]["scale"] = (
                        torch.ones(motion.shape[0], device=dist_util.dev()) * scale
                    )

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                # model.model.clip_model.get_submodule("token_embedding").weight.data
                for t in range(repeat_times):

                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )

                    if t == 0:
                        sub_dicts = [
                            {
                                "motion": sample[bs_i]
                                .squeeze()
                                .permute(1, 0)
                                .cpu()
                                .numpy(),
                                "length": model_kwargs["y"]["lengths"][bs_i]
                                .cpu()
                                .numpy(),
                                "caption": model_kwargs["y"]["text"][bs_i],
                                "tokens": tokens[bs_i],
                                "cap_len": len(tokens[bs_i]),
                                "name": model_kwargs["y"]["name"][bs_i],
                            }
                            for bs_i in range(dataloader.batch_size)
                        ]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [
                            {
                                "motion": sample[bs_i]
                                .squeeze()
                                .permute(1, 0)
                                .cpu()
                                .numpy(),
                                "length": model_kwargs["y"]["lengths"][bs_i]
                                .cpu()
                                .numpy(),
                            }
                            for bs_i in range(dataloader.batch_size)
                        ]

                if is_mm:
                    mm_generated_motions += [
                        {
                            "caption": model_kwargs["y"]["text"][bs_i],
                            "tokens": tokens[bs_i],
                            "cap_len": len(tokens[bs_i]),
                            "mm_motions": mm_motions[
                                bs_i :: dataloader.batch_size
                            ],  # collect all 10 repeats from the (32*10) generated motions
                        }
                        for bs_i in range(dataloader.batch_size)
                    ]

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens, fname = (
            data["motion"],
            data["length"],
            data["caption"],
            data["tokens"],
            data["name"],
        )
        sent_len = data["cap_len"]

        if self.dataset.mode in ["eval", "eval_mti"]:
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (
                denormed_motion - self.dataset.mean_for_eval
            ) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            fname,
        )


class CompMTIGeneratedDataset(Dataset):
    def __init__(
        self,
        model,
        diffusion,
        dataloader,
        mm_num_samples,
        mm_num_repeats,
        max_motion_length,
        num_samples_limit,
        scale=1.0,
        inbetweening_mode=False,  # FIXME
    ):
        self.inbetweening_mode = inbetweening_mode
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        self.default_token_embedding = model.model.clip_model.get_submodule(
            "token_embedding"
        ).weight.data.clone()
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print("real_num_batches", real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(
                real_num_batches,
                mm_num_samples // dataloader.batch_size + 1,
                replace=False,
            )
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print("mm_idxs", mm_idxs)

        from model.tokenizers.CLIPTokenizer import CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path="", placeholder_token="<*>"
        )

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):
                if (
                    num_samples_limit is not None
                    and len(generated_motion) >= num_samples_limit
                ):
                    break

                tokens = [t.split("_") for t in model_kwargs["y"]["tokens"]]

                # add CFG scale to batch
                if scale != 1.0:
                    model_kwargs["y"]["scale"] = (
                        torch.ones(motion.shape[0], device=dist_util.dev()) * scale
                    )

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                #
                _model_kwargs = None
                print("repeat_times: ", repeat_times)
                for t in range(repeat_times):
                    if _model_kwargs is None:
                        _model_kwargs = self.model_kwargs_encoded_appender(
                            model, model_kwargs, dataloader.batch_size
                        )
                    if inbetweening_mode:
                        prefix_end = 0.25
                        suffix_start = 0.75
                        _model_kwargs["y"]["inpainted_motion"] = motion.to(
                            device=dist_util.dev()
                        )
                        _model_kwargs["y"]["inpainting_mask"] = torch.ones_like(
                            motion, dtype=torch.bool, device=dist_util.dev()
                        )  # True means use gt motion

                        for i, length in enumerate(
                            _model_kwargs["y"]["lengths"].cpu().numpy()
                        ):
                            start_idx, end_idx = int(prefix_end * length), int(
                                suffix_start * length
                            )
                            _model_kwargs["y"]["inpainting_mask"][
                                i, :, :, start_idx:end_idx
                            ] = False  # do inpainting in those frames

                            # for vis # FIXME
                            # gt_frames_per_sample[i] = list(range(0, start_idx)) + list(
                            # range(end_idx, max_motion_length)
                            # )

                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=_model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=True,  # for now - understand how much we lost perf
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )

                    if t == 0:
                        sub_dicts = [
                            {
                                "motion": sample[bs_i]
                                .squeeze()
                                .permute(1, 0)
                                .cpu()
                                .numpy(),
                                "length": _model_kwargs["y"]["lengths"][bs_i]
                                .cpu()
                                .numpy(),
                                "caption": _model_kwargs["y"]["text"][bs_i],
                                "tokens": tokens[bs_i],
                                "cap_len": len(tokens[bs_i]),
                                "name": _model_kwargs["y"]["name"][bs_i],
                            }
                            for bs_i in range(dataloader.batch_size)
                        ]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [
                            {
                                "motion": sample[bs_i]
                                .squeeze()
                                .permute(1, 0)
                                .cpu()
                                .numpy(),
                                "length": _model_kwargs["y"]["lengths"][bs_i]
                                .cpu()
                                .numpy(),
                            }
                            for bs_i in range(dataloader.batch_size)
                        ]

                if is_mm:
                    mm_generated_motions += [
                        {
                            "caption": _model_kwargs["y"]["text"][bs_i],
                            "tokens": tokens[bs_i],
                            "cap_len": len(tokens[bs_i]),
                            "mm_motions": mm_motions[
                                bs_i :: dataloader.batch_size
                            ],  # collect all 10 repeats from the (32*10) generated motions
                        }
                        for bs_i in range(dataloader.batch_size)
                    ]

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens, fname = (
            data["motion"],
            data["length"],
            data["caption"],
            data["tokens"],
            data["name"],
        )
        # print(data.keys())
        sent_len = data["cap_len"]

        if self.dataset.mode in ["eval", "eval_mti"]:
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)

            # do it here
            # _denormed_motion, isLoaded = load_motion(fname, denormed_motion)

            # if isLoaded:
            #     denormed_motion = _denormed_motion.cpu().numpy()

            renormed_motion = (
                denormed_motion - self.dataset.mean_for_eval
            ) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            fname,
        )

    """_summary_ input old model and 
                  return new model + kwargs of batch idx if possible
    """

    def unsqueeze_any(self, elem):
        try:
            x = elem.unsqueeze(0)
        except:
            x = [elem]
        return x

    def add_tokenized(self, samp):
        max_text_len = 20
        default_context_length = 77
        context_length = max_text_len + 2
        if type(samp) == tuple:
            raw_text = samp[-1]["y"]["text"]
        elif type(samp) == dict and "y" in samp:
            raw_text = samp["y"]["text"]

        from model.tokenizers.CLIPTokenizer import tokenize

        texts = tokenize(
            self.tokenizer, raw_text, context_length=context_length, truncate=True
        )
        zero_pad = torch.zeros(
            [texts.shape[0], default_context_length - context_length],
            dtype=texts.dtype,
            device=texts.device,
        )
        texts = torch.cat([texts, zero_pad], dim=1)

        if type(samp) == tuple:
            samp[1]["y"]["tokenized"] = texts
        elif type(samp) == dict and "y" in samp:
            samp["y"]["tokenized"] = texts
        return samp

    # def model_kwargs_encoded_appender(self, model, model_kwargs, batchsize):
    #     # to accelerate (avoid alter the model param multiple times)
    #     # we save all first in a map and combine later
    #     device = model.model.clip_model.get_submodule(
    #         "token_embedding"
    #     ).weight.data.device
    #     encoded = {}
    #     name = {}
    #     for b_idx in range(batchsize):
    #         current_name = model_kwargs["y"]["name"][b_idx]
    #         ckpt_exist = (
    #             self.dataset.t2m_dataset.data_dict[current_name]["cparam"] is not None
    #         )  # if we have the cparam

    #         if ckpt_exist:
    #             # rewrite the token embedding weight
    #             model.model.clip_model.get_submodule("token_embedding").weight.data = (
    #                 self.dataset.t2m_dataset.data_dict[current_name]["cparam"]
    #                 .clone()
    #                 .to(device=device)
    #             )

    #             # handle the text too
    #             model_kwargs["y"]["text"][b_idx] = "a person <*>."
    #         else:
    #             # put back the default param of text encoder
    #           model.model.clip_model.get_submodule(
    #               "token_embedding"
    #           ).weight.data = self.default_token_embedding.clone()
    #             # overkill -> tokenized 32 -> but 1 is needed
    #             self.add_tokenized(model_kwargs)

    #             # encoded[b_idx] = model.model.encode_text(
    #             #     model_kwargs["y"]["text"][b_idx]
    #             # )
    #             encoded[b_idx] = model.model.clip_model.encode_text(
    #                 model_kwargs["y"]["tokenized"][b_idx].unsqueeze(0).to(device=device)
    #             ).float()
    #             name[b_idx] = model_kwargs["y"]["name"][b_idx]

    #     # compute for b_idx which do not have cparam
    #     for b_idx in range(batchsize):
    #         if b_idx not in encoded.keys():
    #             encoded[b_idx] = model.model.encode_text(
    #                 model_kwargs["y"]["text"][b_idx]
    #             )
    #             name[b_idx] = model_kwargs["y"]["name"][b_idx]

    #     assert len(encoded) == batchsize

    #     model_kwargs["y"]["encoded"] = torch.stack(
    #         [x[1].squeeze(0) for x in sorted(encoded.items(), key=lambda x: x[0])]
    #     )
    #     # names = torch.stack([x[1] for x in sorted(name.items(), key=lambda x: x[0])])

    #     return model_kwargs

    def model_kwargs_encoded_appender(self, model, model_kwargs, batchsize):
        # to accelerate (avoid alter the model param multiple times)
        # we save all first in a map and combine later
        device = model.model.clip_model.get_submodule(
            "token_embedding"
        ).weight.data.device
        encoded = {}
        name = {}
        for b_idx in range(batchsize):
            ckpt_exist = (
                self.dataset.t2m_dataset.data_dict[model_kwargs["y"]["name"][b_idx]][
                    "cparam"
                ]
                is not None
            )  # if we have the cparam

            if ckpt_exist:

                model.model.clip_model.get_submodule("token_embedding").weight.data = (
                    self.dataset.t2m_dataset.data_dict[
                        model_kwargs["y"]["name"][b_idx]
                    ]["cparam"]
                    .clone()
                    .to(device=device)
                )
                # handle the text too
                model_kwargs["y"]["text"][b_idx] = "a person <*>."
                # overkill -> tokenized 32 -> but 1 is needed
                self.add_tokenized(model_kwargs)
                encoded[b_idx] = model.model.clip_model.encode_text(
                    model_kwargs["y"]["tokenized"][b_idx].unsqueeze(0).to(device=device)
                ).float()
                name[b_idx] = model_kwargs["y"]["name"][b_idx]

        # put back the default param of text encoder
        model.model.clip_model.get_submodule(
            "token_embedding"
        ).weight.data = self.default_token_embedding.clone()

        for b_idx in range(batchsize):
            if b_idx not in encoded.keys():
                encoded[b_idx] = model.model.encode_text(
                    model_kwargs["y"]["text"][b_idx]
                )

        assert len(encoded) == batchsize

        # _name = [name[x] for x in sorted(name)]

        # model_kwargs["y"]["encoded"] = torch.stack(
        #     [x[1].squeeze(0) for x in sorted(encoded.items(), key=lambda x: x[0])]
        # )

        model_kwargs["y"]["encoded"] = torch.stack(
            [encoded[x].squeeze(0) for x in sorted(encoded)]
        )
        # import ipdb

        # ipdb.set_trace()
        return model_kwargs
