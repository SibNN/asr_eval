asr_eval
###################

This Python project is a set of tools for:

1. Evaluating and load testing streaming ASR (automatic speech recognition)
2. Evaluating multivariant-labeled ASR

It can also serve as a streaming inference wrapper and a collection
of ASR models and datasets (with focus on Russian speech).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

The main features:

Feature overview
======================


Multivariant string alignment
********************************

**Problem**. Often a speech can be transcribed in multiple ways, such as *"On the {seventh|7} - {eighth|8} of May, the {sixteenth|16th|16-th} stage of Formula {1|One} was held."* A usual stage of text normalization does not handle all cases well, especially for non-latin languages with rich word formation. Also, some parts of speech are poorly heard: the longer the audio, the more likely it is to contain such sections, so the problem is especially relevant for a long-form transcription. Such parts can be annotathed by `<*>` ("Anything"): a symbol that matches any word sequence, possibly empty.

**Solution**. Multivariant string alignment with "Anything" insertions is implemented. This allows you to create high-quality datasets for ASR evaluation.

Enhanced streaming string alignment
*************************************

**Problem**. Calculating WER (word error rate) requires to align strings, but there are many WER-optimal alignments. Let the speaker say *"Multivariant... one, two, three"* which was transcribed as *"Multivariate"*. One of WER-optimal alignments aligns *"Multivariate"* with *"three"* (a replacement), and other words with nothing (deletions). This gives the false conclusion that the word *"three"* was already transribed incorrectly. This will lead to incorrect calculation of the latency metric (current time minus last transcribed word time).

**Solution**. Among all the WER-optimal alignments we search by several other criteria, such as CER-optimality. This gives higher quality alignments, both for streaming and non-streaming ASR.


Preparing streaming datasets
********************************

**Word timings**. An algorithm is impelemented to perform force alignment using CTC models, such as GigaAM 2, to obtain timings for each word, also for multivariant labeling. Such a pseudo-labeling gives gives an error of about 0.2, rarely up to 0.5 seconds, that is enough to test streaming ASR latency and partial transcription quality.

Preparing streaming models
********************************

**StreamingASR**. An abstract class that starts a separate thread, accepts input chunks (audio floats, ints or bytes) and returns output chunks (transcription). To implement a concrete model, subclass `StreamingASR` and implement the method that waits for input chunks and returns output chunks. Each chunk is supplemented with a unqiue identifier, which allows to transcribe multiple audios in parallel.

**Rechunking**. As some models may be sensitive to a chunk size, `StreamingASR` allows to wait until the data of required size become available.

**Editing transcription**. Each output chunk has a unique ID that can be overwritten by the future chunks. This allows model to edit the previously transcribed words.

**Senders**. A utility class for testing that accepts a full waveform, splits it and sends audio chunks at the required speed.

**Full history**. After transcribing, a full history of input and output chunks is available, with timings. They can be serialized to json and back. This allows to inpect model behaviour and partial results, and create diagrams.

Evaluating streaming models
********************************

**Partial alignment diagram**. Given a history of input and output chunks, we perform multiple partial alignments and draw a rich diagram that shows how does the model makes and corrects a prediction over time. This reveals various inconspicuous problems in the functioning of the model.

**Disentangling latency sources**. Models return the length of the processed audio along with the transcription. This allows to disentangle two sources of latency: the model refuses to predict until sufficient right context is received, and the model can't process the input stream in time.

**Aggregated diagrams**. We provide a set of diagrams to summarize the model's behavior over the entire dataset.

**Densified mode**. Evaluating models in real time can cause situations where both sender and model waits, if the model processes the stream faster than real time. To speed up, we implement a special mode where we send all the input chunks at once, allow the model to process them sequentially, and then remap timings to simulate real-time receipt. Note that this is expected to work only for some models.

Deploying streaming models
********************************

The `StreamingASR` class unifies the behaviour of any streaming ASR model and is a ready-to-use wrapper for production deployment.