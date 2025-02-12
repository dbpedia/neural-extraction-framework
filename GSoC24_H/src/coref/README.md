
All the scripts are primarily taken from https://github.com/vdobrovolskii/wl-coref

Some changes we made to run the model (train+inference) on TransMuCores data

Modified config.toml and pairwise_encoder.py

Added line 65 in the coref_model.py (self.config.device = ...)

We have turned off the self._build_optimizers() (comment out the line) in CorefModel because we intent to run the model in inference mode.

