# Sentiment-Classification-using-RNN-based-models

In this repository, you will find code and resources for using Recurrent Neural Network (RNN) based models to classify text data for sentiment analysis. Sentiment analysis is a common task in natural language processing, where the goal is to determine the emotional tone of a piece of text. This can be useful for a wide range of applications, such as analyzing customer reviews to understand the overall sentiment towards a product, or analyzing social media posts to understand public sentiment on a particular topic.

The code in this repository includes examples of how to train and evaluate RNN-based models for sentiment classification using various datasets and libraries such as Pytorch.

# Usage
To use the script to train the model for sentiment classification, please run the below code:
```bash
python main.py --model=RNN --inputfile=data/crowdflower_emotion.csv
```
Where,
- `--model`: `RNN` or `LSTM`. It is the model choice between RNN and LSTM.
- `--inputfile`: `Optional`. Path to the input csv file having first column as sentiment class and second column as text.

I hope that this repository will be a useful resource for those looking to learn more about using RNNs for sentiment analysis and natural language processing tasks. Please feel free to reach out with any questions or suggestions, and we look forward to seeing what you build with these tools!
