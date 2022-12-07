from .model import COCML, HarCML


# note that if k == 1, then AJCML == CML
# SoftCML levrages softmin to norm and selectsx the k-th embeddings 
__all__ = ['COCML', 'HarCML']