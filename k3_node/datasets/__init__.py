from .karate import KarateClub
from .fake import FakeDataset, FakeHeteroDataset
from .planetoid import Planetoid
from .tu_dataset import TUDataset
from .citation_full import CitationFull, CoraFull
from .amazon import Amazon
from .coauthor import Coauthor
from .wikics import WikiCS
from .webkb import WebKB
from .actor import Actor
from .polblogs import PolBlogs
from .airports import Airports
from .email_eu_core import EmailEUCore
from .github import GitHub
from .facebook import FacebookPagePage
from .lastfm_asia import LastFMAsia
from .twitch import Twitch
from .ba_shapes import BAShapes
from .ba2motif_dataset import BA2MotifDataset
from .sbm_dataset import StochasticBlockModelDataset, RandomPartitionGraphDataset
from .explainer_dataset import ExplainerDataset
from .entities import Entities
from .word_net import WordNet18, WordNet18RR
from .freebase import FB15k_237
from .dblp import DBLP
from .imdb import IMDB
from .qm7 import QM7b
from .molecule_net import MoleculeNet

__all__ = [
    "KarateClub",
    "FakeDataset",
    "FakeHeteroDataset",
    "Planetoid",
    "TUDataset",
    "CitationFull",
    "CoraFull",
    "Amazon",
    "Coauthor",
    "WikiCS",
    "WebKB",
    "Actor",
    "PolBlogs",
    "Airports",
    "EmailEUCore",
    "GitHub",
    "FacebookPagePage",
    "LastFMAsia",
    "Twitch",
    "BAShapes",
    "BA2MotifDataset",
    "StochasticBlockModelDataset",
    "RandomPartitionGraphDataset",
    "ExplainerDataset",
    "Entities",
    "WordNet18",
    "WordNet18RR",
    "FB15k_237",
    "DBLP",
    "IMDB",
    "QM7b",
    "MoleculeNet",
]

