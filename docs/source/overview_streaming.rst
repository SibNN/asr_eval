Streaming features
###############################

Preparing streaming datasets
********************************

**Word timings**. An algorithm is impelemented to perform force alignment using CTC models, such as GigaAM 2, to obtain timings for each word, also for multivariant labeling. Such a pseudo-labeling gives gives an error of about 0.2, rarely up to 0.5 seconds, that is enough to test streaming ASR latency and partial transcription quality.

.. _Preparing streaming models:

Preparing streaming models
********************************

We provide a set of ready to use model wrappers for streaming speech recognition, wrapped in a unified interface.

**StreamingASR**. An abstract class that starts a separate thread, accepts input chunks (audio floats, ints or bytes) and returns output chunks (transcription). To implement a concrete model, subclass `StreamingASR` and implement the method that waits for input chunks and returns output chunks. Each chunk is supplemented with a unqiue identifier, which allows to transcribe multiple audios in parallel.

**Rechunking**. As some models may be sensitive to a chunk size, `StreamingASR` allows to wait until the data of required size become available.

**Editing transcription**. Each output chunk has a unique ID that can be overwritten by the future chunks. This allows model to edit the previously transcribed words.

**Senders**. A utility class for testing that accepts a full waveform, splits it and sends audio chunks at the required speed.

**Full history**. After transcribing, a full history of input and output chunks is available, with timings. They can be serialized to json and back. This allows to inpect model behaviour and partial results, and create diagrams.

Evaluating streaming models
********************************

We provide tools to evaluate streaming speech recognition and make informative diagrams.

**Partial alignment diagram**. Given a history of input and output chunks, we perform multiple partial alignments and draw a rich diagram that shows how does the model makes and corrects a prediction over time. This reveals various inconspicuous problems in the functioning of the model.

**Aggregated diagrams**. We provide a set of diagrams to summarize the model's behavior over the entire dataset.

**Disentangling latency sources**. Models return the length of the processed audio along with the transcription. This allows to disentangle two sources of latency: the model refuses to predict until sufficient right context is received, and the model can't process the input stream in time.

**Densified mode**. Evaluating models in real time can cause situations where both sender and model waits, if the model processes the stream faster than real time. To speed up, we implement a special mode where we send all the input chunks at once, allow the model to process them sequentially, and then remap timings to simulate real-time receipt. Note that this is expected to work only for some models.

Deploying streaming models
********************************

The `StreamingASR` class unifies the behaviour of any streaming ASR model and is a ready-to-use wrapper for production deployment.