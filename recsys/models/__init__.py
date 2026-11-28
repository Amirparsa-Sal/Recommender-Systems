"""Importing this package registers every model in :data:`MODEL_REGISTRY`."""

from recsys.models.base import MODEL_REGISTRY, Recommender, create_model, get_model_class
from recsys.models.popular import TopPopular
from recsys.models.knn import ItemKNN, UserKNN
from recsys.models.slim_elasticnet import SLIMElasticNet
from recsys.models.slim_bpr import SLIMBPR
from recsys.models.als import MFALS
from recsys.models.funk_svd import FunkSVD
from recsys.models.asymmetric_svd import AsymmetricSVD
from recsys.models.bpr_mf import BPRMF
from recsys.models.autoencoder import Autoencoder, DenoisingAutoencoder, MultVAE
from recsys.models.ease import EASE
from recsys.models.gfcf import GFCF
from recsys.models.lightgcn import LightGCN
from recsys.models.graph import P3Alpha, RP3Beta, RandomWalk, RandomWalkWithRestart
from recsys.models.hybrid import HybridLinear, HybridList, HybridMerge, HybridPipeline

__all__ = [
    "MODEL_REGISTRY",
    "Recommender",
    "create_model",
    "get_model_class",
    "TopPopular",
    "ItemKNN",
    "UserKNN",
    "SLIMElasticNet",
    "SLIMBPR",
    "MFALS",
    "FunkSVD",
    "AsymmetricSVD",
    "BPRMF",
    "EASE",
    "GFCF",
    "LightGCN",
    "Autoencoder",
    "DenoisingAutoencoder",
    "MultVAE",
    "RandomWalk",
    "RandomWalkWithRestart",
    "P3Alpha",
    "RP3Beta",
    "HybridLinear",
    "HybridList",
    "HybridPipeline",
    "HybridMerge",
]
