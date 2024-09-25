# Unveiling Influential Factors in Classifying Domain Entities into Top-Level Ontology Concepts

This repository contains the code and datasets for the paper **"Unveiling Influential Factors in Classifying Domain Entities into Top-Level Ontology Concepts: An Analysis Using GO and ChEBI Ontologies"** presented at ONTOBRAS 2024.

## Overview

In the realm of ontology engineering, accurately classifying domain entities into top-level ontology concepts is crucial. This repository explores the influential factors affecting the performance of using informal definitions to represent domain entities textually, leveraging Language Models to generate embedding vectors, and employing the K-Nearest Neighbors (KNN) algorithm to classify these embeddings into top-level ontology concepts.

The study particularly focuses on the Gene Ontology (GO) and Chemical Entities of Biological Interest (ChEBI) ontologies. We hypothesize that the embedding representation of informal definitions from highly specialized domains might present different behaviors regarding their proximity with other informal definitions of other domains, influencing the predicted top-level ontology concept.

## Key Findings

- The relationship between the proximity of domain entities in the embedding space and their top-level ontology concepts varies according to domain specificity.
- The classifier's performance is strongly influenced by how ontology developers write informal definitions within each domain.
- The study underscores the potential of informal definitions in reflecting top-level ontology concepts and suggests using consolidated domain entities in a domain ontology during the classifier's training stage.

## Repository Structure

- `log/`: Contains the output results of all experiments considering all domains and domain ontologies.
- `datasets/`: Includes the datasets used for the experiments, with entities from the GO and ChEBI ontologies.
- `confusion matrix/`: Contains the output confusion matrixes of all experiments considering all domains and domain ontologies.
- The rest of the folder contains the required packages to perform the experiments
## Installation

To replicate the experiments, clone the repository and install the required dependencies:

```bash
git clone https://github.com/BDI-UFRGS/Alcides-ONTOBRAS-2024.git
cd Alcides-ONTOBRAS-2024
pip install -r requirements.txt
