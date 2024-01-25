import os
import time
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from actor import CustomActorModule
from trainer import CustomTrainer
from tmrl.networking import Trainer, RolloutWorker, Server
from tmrl.training_offline import TrainingOffline
from tmrl.util import partial
from utils import parse_args

os.environ['WANDB_API_KEY'] = cfg.WANDB_KEY

memory = partial(
    cfg_obj.MEM,
    memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
    batch_size=cfg.TMRL_CONFIG["BATCH_SIZE"],
    sample_preprocessor=None,
    dataset_path=cfg.DATASET_PATH,
    imgs_obs=cfg.IMG_HIST_LEN,
    act_buf_len=cfg.ACT_BUF_LEN,
    crc_debug=False)

custom_trainer = partial(CustomTrainer)

training_offline = partial(
    TrainingOffline,
    env_cls=cfg_obj.ENV_CLS,
    memory_cls=memory,
    training_agent_cls=custom_trainer,
    device='cuda' if cfg.CUDA_TRAINING else 'cpu',
    epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],
    steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],
    rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],
    update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],
    update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],
    start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"],
    max_training_steps_per_env_step=cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"])

if __name__ == "__main__":
    args = parse_args()

    if args.s:
        server = Server(
            port=cfg.PORT,
            password=cfg.PASSWORD,
            security=cfg.SECURITY)

        while True:
            time.sleep(0.5)
    elif args.t:
        trainer = Trainer(
            training_cls=training_offline,
            server_ip=cfg.SERVER_IP_FOR_TRAINER,
            server_port=cfg.PORT,
            password=cfg.PASSWORD,
            security=cfg.SECURITY)

        trainer.run_with_wandb(
            run_id=cfg.WANDB_RUN_ID,
            entity=cfg.WANDB_ENTITY,
            project=cfg.WANDB_PROJECT)
    else:
        rollout_worker = RolloutWorker(
            env_cls=cfg_obj.ENV_CLS,
            actor_module_cls=CustomActorModule,
            device='cpu',
            sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
            server_ip=cfg.SERVER_IP_FOR_WORKER,
            server_port=cfg.PORT,
            security=cfg.SECURITY,
            password=cfg.PASSWORD,
            max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
            obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
            standalone=args.a is None)

        rollout_worker.run()
