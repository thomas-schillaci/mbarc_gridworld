class Logger:

    def __init__(self, config, world_model):
        self.active = config.use_wandb
        if self.active:
            import wandb
            wandb.init(project='MBARC (gridworld)', name=config.experiment_name, config=config, reinit=True)
            wandb.watch(world_model)

    def log(self, *args, **kwargs):
        if self.active:
            import wandb
            wandb.log(*args, **kwargs)
