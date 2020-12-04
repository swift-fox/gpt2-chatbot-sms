# A GPT-2 Chatbot

This is a simple and crappy chatbot built on top of the groundbreaking OpenAI GPT-2 transformer language model.

## Demo

Checkout [chatbot.swift-fox.com](https://chatbot.swift-fox.com), or text "hello" to `205-671-2142` to chat with it if you are in the US.

It can talk like a real person for a while. But soon you'll discover the real crappiness of it. So hope you'll be amused. To restart a chat, refresh the page or text "/reset". In each chat, it impersonates a different character. It has 17878 personalities. A peculiar case of [DID](https://en.wikipedia.org/wiki/Dissociative_identity_disorder).

<img src="https://github.com/swift-fox/gpt2-chatbot-sms/raw/master/images/chatbot-demo-1.png" width="30%"/>

If the website is down, text messaging won't work either. That means I've decided to stop spending $15 a month to amuse the world.

## How it works

### The Language Model

The model at work is [DistilGPT2](https://huggingface.co/distilgpt2) by Hugging Face. It's a distilled version of the OpenAI GPT-2 model. It has 82 million parameters comparing to 124 million of the smallest GPT-2 model.

The model is put into a text generation pipeline to generate replies to function as a chatbot. However, this is a much different task than regular text generation. Chat replies are short and chatty. The corpus is different from what GPT-2 was trained on. So a retraining (fine-tuning) of the model is required.

On the decoding side, it uses top-k top-p filtering. It's easier to implement and performs better than beam search.

### Training

The dataset to retrain (fine-tune) the model is PersonaChat by Facebook. I'm using the [copy](https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json) preprocessed by Hugging Face.

The training code is probably the most unsubtle one on earth to train a transformer. But it works anyway. It took 9 hours to train on a NVIDIA GTX 1080 video card from 2016 with 8 GB memory. The minuscule memory limits the batch size to 2, which made the training so slow.

### Deployment

The model and the demo are deployed to a virtual machine in Google Cloud with 2 vCPU cores and 2 GB memory. So the inference runs on the CPU, which is fine for a demo. And 2 GB memory is the minimal to serve a small GPT-2 model. This setup can generate ~10 words per second. So it's definitely a small-scale prototype. The total cost to run it is about $15 a month including the VM, the domain name, and the Twilio API.

Attempts were made to deploy it to App Engine, which ended up badly because of a 7-second cold start delay and some nasty limitations.

### Known Issues

1. The majority of its crappiness came from the fact that only 5% of the dataset was used to train the model. The other 95% can be utilized with a different training task which is to distinguish between distractors from the ground truth. This can be accomplished by multi-task training with an additional prediction head to the model and setting proper coefficients to the two training losses.

1. An unoptimized TensorFlow model is definitely not ideal for deployment. It loads slow and runs slow. A smarter person would at least try some quantization and converting it to the ONNX format. There is a tool in the transformers library to do so. But it doesn't really work for this case. So some research is required for this optimization. Once done, the model has the potential to be deployed to App Engine to further reduce the hosting cost.

## Credits

Many thanks to Hugging Face for their fantastic [transformers library](https://github.com/huggingface/transformers) and [the article](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313) about their work on the conversational AI.
