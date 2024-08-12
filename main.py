from dataset.dlp import DLPDatasetBuilder
from dataset.bfo import BFODatasetBuilder
from encoder.embedding import EmbeddingGenerator
from experiment.per_ontology import PerOntologyToTLP, PerFileExperiment
from experiment.per_tlp import PerTLPExperiment
from experiment.per_domain import PerDomainExperiment
from experiment.cross_domain import CrossDomainExperiment
from experiment.cross_tlp import CrossTLPExperiment

dlp_target_classes =  ['abstract', 'endurant', 'perdurant', 'quality']
bfo_target_classes =  ['independent continuant',
                       'specifically dependent continuant',
                       'generically dependent continuant',  
                       'process']

bfo_domains = ['agriculture',
              'biological systems',
              'chemistry and biochemistry',
              'diet, metabolomics, and nutrition',
              'environment',
              'health',
              'information',
              'investigations',
              'microbiology',
              'organisms']

dlp_domains = ['WN', 'WIKI']



# DLPDatasetBuilder(file_dir='datasets\\DLP\WIKI.csv', output_dir='input_dataset\\dlp', target_classes=dlp_target_classes)
# DLPDatasetBuilder(file_dir='datasets\\DLP\WN.csv', output_dir='input_dataset\\dlp', target_classes=dlp_target_classes)
BFODatasetBuilder(folder_dir='datasets\\BFO', output_dir='input_dataset\\bfo', target_classes=bfo_target_classes)


# EmbeddingGenerator(input_dir='input_dataset\\bfo', output_dir='embedding_dataset\\bfo')
# EmbeddingGenerator(input_dir='input_dataset\\dlp', output_dir='embedding_dataset\\dlp')


# PerFileExperiment(tlp='bfo', domain='biological systems', dataset='go')
# PerFileExperiment(tlp='bfo', domain='chemistry and biochemistry', dataset='chebi')
# PerDomainExperiment(tlp='dlp', domain='WN')
# PerTLPExperiment(tlp='bfo')


# for domain in bfo_domains:
#     PerDomainExperiment(tlp='bfo', domain=domain)

# for domain in dlp_domains:
#     PerDomainExperiment(tlp='dlp', domain=domain)


# CrossDomainExperiment(tlp='bfo')


# PerOntologyToTLP(tlp='bfo')

# from plot.bar_chat import Bar


# Bar()

# CrossTLPExperiment(train_tlp='dlp', test_tlp='bfo')

# file_path = 'embedding_dataset\dlp\wn.h5'
# with h5py.File(file_path, 'r') as h5_file:
#     # List all groups
#     print("Keys: %s" % h5_file.keys())
    
#     # Get the data
#     dataset_name = 'embeddings'  # Replace with your dataset name
#     data = h5_file[dataset_name][:]
#     print(len(data))