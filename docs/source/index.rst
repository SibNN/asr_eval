asr_eval
###################

Evaluation and building components for Automatic Speech Recognition (ASR) in Python.

See the `preprint paper <https://arxiv.org/abs/2601.20992>`_ and full PDF documentation.

* Evaluating ASR with extended annotation syntax
* Running and evaluating non-streaming ASR, and perform a detailed error analysis
* Running and evaluating streaming ASR, producing many informative diagrams

.. grid:: 1
    :margin: 0
    :class-row: suppress-margin-bottom

    .. grid-item:: 
        :columns: 10

        .. card:: 🌟 Feature overview
           :text-align: center
           :shadow: none
           :margin: 0
           :class-card: noborder

.. grid:: 2

    .. grid-item-card::  
        :columns: 5
        :link: overview_alignment
        :link-type: doc
        :class-body: index-page-card

        Alignment features

    .. grid-item-card::  
        :columns: 5
        :link: overview_streaming
        :link-type: doc
        :class-body: index-page-card

        Streaming features

.. grid:: 1
    :margin: 0
    :class-row: suppress-margin-bottom

    .. grid-item:: 
        :columns: 10

        .. card:: 🛠️ Guides
           :text-align: center
           :shadow: none
           :margin: 0
           :class-card: noborder

.. grid:: 2

    .. grid-item-card::  
        :columns: 5
        :link: guide_installation
        :link-type: doc
        :class-body: index-page-card

        Installation

    .. grid-item-card::  
        :columns: 5
        :link: guide_alignment_wer
        :link-type: doc
        :class-body: index-page-card

        Alignments and WER

.. grid:: 2

    .. grid-item-card::  
        :columns: 5
        :link: guide_streaming_evaluation
        :link-type: doc
        :class-body: index-page-card

        Streaming evaluation

    .. grid-item-card::  
        :columns: 5
        :link: guide_evaluation_dashboard
        :link-type: doc
        :class-body: index-page-card

        Evaluation and dashboard

.. grid:: 1
    :margin: 0
    :class-row: suppress-margin-bottom

    .. grid-item:: 
        :columns: 10

        .. card:: 📚 Components
           :text-align: center
           :shadow: none
           :margin: 0
           :class-card: noborder

.. grid:: 2

    .. grid-item-card::  
        :columns: 5
        :link: guide_models_list
        :link-type: doc
        :class-body: index-page-card

        Models list

    .. grid-item-card::  
        :columns: 5
        :link: guide_datasets_list
        :link-type: doc
        :class-body: index-page-card

        Datasets list

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Overview

   asr_eval <self>
   overview_alignment
   overview_streaming

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Guides

   guide_installation
   guide_alignment_wer
   guide_streaming_evaluation
   guide_evaluation_dashboard
   guide_datasets_list
   guide_models_list

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference

   asr_eval.align
   asr_eval.streaming
   asr_eval.models
   asr_eval.bench
   asr_eval.ctc
   asr_eval.normalizing
   asr_eval.linguistics
   asr_eval.correction
   asr_eval_other

Acknowledgements
===================

Several components of this project, related to the :mod:`~asr_eval.correction` module, were developed in
collaboration with Yandex Camp students during July 2025:

* Timur Rafikov https://github.com/timur-rafikov
* Vasily Kudryavtsev https://github.com/SLENSER0
* Yana Fitkovskaja https://github.com/fitkovskaja
* Dmitry Ezhov https://github.com/EZHOWWW
* Nikita Vambrikov https://github.com/Yourkrash
* Anastasia Zvereva https://github.com/zvrva
* Valeria Stolyarova https://github.com/lerastol
* Mukharyam Baviev https://github.com/Mukharyam
* Antal Schiff https://github.com/schiff-heicho
* Mirzomansurkhon Sultanov https://github.com/Ultimatereo