---
license: openrail
widget:
- text: I am totally a human, trust me bro.
  example_title: default
- text: >-
    In Finnish folklore, all places and things, and also human beings, have a
    haltija (a genius, guardian spirit) of their own. One such haltija is called
    etiäinen—an image, doppelgänger, or just an impression that goes ahead of a
    person, doing things the person in question later does. For example, people
    waiting at home might hear the door close or even see a shadow or a
    silhouette, only to realize that no one has yet arrived. Etiäinen can also
    refer to some kind of a feeling that something is going to happen. Sometimes
    it could, for example, warn of a bad year coming. In modern Finnish, the
    term has detached from its shamanistic origins and refers to premonition.
    Unlike clairvoyance, divination, and similar practices, etiäiset (plural)
    are spontaneous and can't be induced. Quite the opposite, they may be
    unwanted and cause anxiety, like ghosts. Etiäiset need not be too dramatic
    and may concern everyday events, although ones related to e.g. deaths are
    common. As these phenomena are still reported today, they can be considered
    a living tradition, as a way to explain the psychological experience of
    premonition.
  example_title: real wikipedia
- text: >-
    In Finnish folklore, all places and things, animate or inanimate, have a
    spirit or "etiäinen" that lives there. Etiäinen can manifest in many forms,
    but is usually described as a kind, elderly woman with white hair. She is
    the guardian of natural places and often helps people in need. Etiäinen has
    been a part of Finnish culture for centuries and is still widely believed in
    today. Folklorists study etiäinen to understand Finnish traditions and how
    they have changed over time.
  example_title: generated wikipedia
- text: >-
    This paper presents a novel framework for sparsity-certifying graph
    decompositions, which are important tools in various areas of computer
    science, including algorithm design, complexity theory, and optimization.
    Our approach is based on the concept of "cut sparsifiers," which are sparse
    graphs that preserve the cut structure of the original graph up to a certain
    error bound. We show that cut sparsifiers can be efficiently constructed
    using a combination of spectral techniques and random sampling, and we use
    them to develop new algorithms for decomposing graphs into sparse subgraphs.
  example_title: from ChatGPT
- text: >-
    Recent work has demonstrated substantial gains on many NLP tasks and
    benchmarks by pre-training on a large corpus of text followed by fine-tuning
    on a specific task. While typically task-agnostic in architecture, this
    method still requires task-specific fine-tuning datasets of thousands or
    tens of thousands of examples. By contrast, humans can generally perform a
    new language task from only a few examples or from simple instructions -
    something which current NLP systems still largely struggle to do. Here we
    show that scaling up language models greatly improves task-agnostic,
    few-shot performance, sometimes even reaching competitiveness with prior
    state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an
    autoregressive language model with 175 billion parameters, 10x more than any
    previous non-sparse language model, and test its performance in the few-shot
    setting. For all tasks, GPT-3 is applied without any gradient updates or
    fine-tuning, with tasks and few-shot demonstrations specified purely via
    text interaction with the model. GPT-3 achieves strong performance on many
    NLP datasets, including translation, question-answering, and cloze tasks, as
    well as several tasks that require on-the-fly reasoning or domain
    adaptation, such as unscrambling words, using a novel word in a sentence, or
    performing 3-digit arithmetic. At the same time, we also identify some
    datasets where GPT-3's few-shot learning still struggles, as well as some
    datasets where GPT-3 faces methodological issues related to training on
    large web corpora. Finally, we find that GPT-3 can generate samples of news
    articles which human evaluators have difficulty distinguishing from articles
    written by humans. We discuss broader societal impacts of this finding and
    of GPT-3 in general.
  example_title: GPT-3 paper
datasets:
- NicolaiSivesind/human-vs-machine
- gfissore/arxiv-abstracts-2021
language:
- en
pipeline_tag: text-classification
tags:
- mgt-detection
- ai-detection
---

Machine-generated text-detection by fine-tuning of language models
===

This project is related to a bachelor's thesis with the title "*Turning Poachers into Gamekeepers: Detecting Machine-Generated Text in Academia using Large Language Models*" (see [here](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3078096)) written by *Nicolai Thorer Sivesind* and *Andreas Bentzen Winje* at the *Department of Computer Science* at the *Norwegian University of Science and Technology*.

It contains text classification models trained to distinguish human-written text from text generated by language models like ChatGPT and GPT-3. The best models were able to achieve an accuracy of 100% on real and *GPT-3*-generated wikipedia articles (4500 samples), and an accuracy of 98.4% on real and *ChatGPT*-generated research abstracts (3000 samples).

The dataset card for the dataset that was created in relation to this project can be found [here](https://huggingface.co/datasets/NicolaiSivesind/human-vs-machine).

**NOTE**: the hosted inference on this site only works for the RoBERTa-models, and not for the Bloomz-models. The Bloomz-models otherwise can produce wrong predictions when not explicitly providing the attention mask from the tokenizer to the model for inference. To be sure, the [pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines)-library seems to produce the most consistent results.


## Fine-tuned detectors

This project includes 12 fine-tuned models based on the RoBERTa-base model, and three sizes of the bloomz-models. 

| Base-model | RoBERTa-base                                                                   | Bloomz-560m                                                                                | Bloomz-1b7                                                                               | Bloomz-3b                                                                              |
|------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Wiki       | [roberta-wiki](https://huggingface.co/andreas122001/roberta-wiki-detector)     | [Bloomz-560m-wiki](https://huggingface.co/andreas122001/bloomz-560m-wiki-detector)         | [Bloomz-1b7-wiki](https://huggingface.co/andreas122001/bloomz-1b7-wiki-detector)         | [Bloomz-3b-wiki](https://huggingface.co/andreas122001/bloomz-3b-wiki-detector)         |
| Academic   | [roberta-academic](https://huggingface.co/andreas122001/roberta-academic-detector) | [Bloomz-560m-academic](https://huggingface.co/andreas122001/bloomz-560m-academic-detector) | [Bloomz-1b7-academic](https://huggingface.co/andreas122001/bloomz-1b7-academic-detector) | [Bloomz-3b-academic](https://huggingface.co/andreas122001/bloomz-3b-academic-detector) |
| Mixed      | [roberta-mixed](https://huggingface.co/andreas122001/roberta-mixed-detector)   | [Bloomz-560m-mixed](https://huggingface.co/andreas122001/bloomz-560m-mixed-detector)       | [Bloomz-1b7-mixed](https://huggingface.co/andreas122001/bloomz-1b7-mixed-detector)       | [Bloomz-3b-mixed](https://huggingface.co/andreas122001/bloomz-3b-mixed-detector)       |


### Datasets

The models were trained on selections from the [GPT-wiki-intros]() and [ChatGPT-Research-Abstracts](), and are separated into three types, **wiki**-detectors, **academic**-detectors and **mixed**-detectors, respectively.

- **Wiki-detectors**:
  - Trained on 30'000 datapoints (10%) of GPT-wiki-intros.
  - Best model (in-domain) is Bloomz-3b-wiki, with an accuracy of 100%.
- **Academic-detectors**:
  - Trained on 20'000 datapoints (100%) of ChatGPT-Research-Abstracts.
  - Best model (in-domain) is Bloomz-3b-academic, with an accuracy of 98.4%
- **Mixed-detectors**:
  - Trained on 15'000 datapoints (5%) of GPT-wiki-intros and 10'000 datapoints (50%) of ChatGPT-Research-Abstracts.
  - Best model (in-domain) is RoBERTa-mixed, with an F1-score of 99.3%.


### Hyperparameters

All models were trained using the same hyperparameters:

```python
{
 "num_train_epochs": 1,
 "adam_beta1": 0.9,
 "adam_beta2": 0.999,
 "batch_size": 8,
 "adam_epsilon": 1e-08
 "optim": "adamw_torch" # the optimizer (AdamW)
 "learning_rate": 5e-05, # (LR)
 "lr_scheduler_type": "linear", # scheduler type for LR
 "seed": 42, # seed for PyTorch RNG-generator.
}
```

### Metrics

Metrics can be found at https://wandb.ai/idatt2900-072/IDATT2900-072.


In-domain performance of wiki-detectors:

| Base model  | Accuracy | Precision | Recall | F1-score |
|-------------|----------|-----------|--------|----------|
| Bloomz-560m | 0.973    | *1.000    | 0.945  | 0.972    |
| Bloomz-1b7  | 0.972    | *1.000    | 0.945  | 0.972    |
| Bloomz-3b   | *1.000   | *1.000    | *1.000 | *1.000   |
| RoBERTa     | 0.998    | 0.999     | 0.997  | 0.998    |


In-domain peformance of academic-detectors:

| Base model  | Accuracy | Precision | Recall | F1-score |
|-------------|----------|-----------|--------|----------|
| Bloomz-560m | 0.964    | 0.963     | 0.965  | 0.964    |
| Bloomz-1b7  | 0.946    | 0.941     | 0.951  | 0.946    |
| Bloomz-3b   | *0.984   | *0.983    | 0.985  | *0.984   |
| RoBERTa     | 0.982    | 0.968     | *0.997 | 0.982    |


F1-scores of the mixed-detectors on all three datasets:

| Base model  | Mixed  | Wiki   | CRA    |
|-------------|--------|--------|--------|
| Bloomz-560m | 0.948  | 0.972  | *0.848 |
| Bloomz-1b7  | 0.929  | 0.964  | 0.816  |
| Bloomz-3b   | 0.988  | 0.996  | 0.772  |
| RoBERTa     | *0.993 | *0.997 | 0.829  |


## Credits

- [GPT-wiki-intro](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro), by Aaditya Bhat
- [arxiv-abstracts-2021](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021), by Giancarlo
- [Bloomz](bigscience/bloomz), by BigScience
- [RoBERTa](https://huggingface.co/roberta-base), by Liu et. al.


## Citation

Please use the following citation:

```
@misc {sivesind_2023,
    author       = { {Nicolai Thorer Sivesind} and {Andreas Bentzen Winje} },
    title        = { Machine-generated text-detection by fine-tuning of language models },
    url          = { https://huggingface.co/andreas122001/roberta-academic-detector },
    year         = 2023,
    publisher    = { Hugging Face }
}
```