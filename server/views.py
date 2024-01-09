from django.shortcuts import render
from django.http import JsonResponse

from typing import Dict, Text
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import os

# Create your views here.
info_dict = {
    'team': 'MonkeyPlayin',
    'members': ['Robert', 'Denis', 'Mihail'],
    'technologies': ['Django', 'Angular', 'Scikit', 'Firestore', 'Docker'],
    'description': 'This API\'s purpose is to process and serve Game Recommendation data!',
    'data_format': 'JSON',
    'request_lib': 'axios',
    'request_example': 'axios.get(\'https://api-monkeyplayin.onrender.com/your_endpoint\').then(res => console.log(res.data))',
}

def info(request):
    return JsonResponse(info_dict)

class GamesModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(
      self,
      games_model: tf.keras.Model,
      genres_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    self.games_model = games_model
    self.genres_model = genres_model

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.

    games_embeddings = self.games_model(features["Title"])
    genres_embeddings = self.genres_model(features["Genres"])

    return self.task(games_embeddings, genres_embeddings)
  
def games_model():
   cdir = os.getcwd()
   dir = os.path.join(cdir, 'server/models')
   filepath = os.path.join(dir, 'games_model')
   loaded_games_model = tf.keras.models.load_model(filepath)
   return loaded_games_model

def genres_model():
    cdir = os.getcwd()
    dir = os.path.join(cdir, 'server/models')
    filepath = os.path.join(dir, 'games_model')
    loaded_genres_model = tf.keras.models.load_model(filepath)
    return loaded_genres_model

def games_tensor():
    cdir = os.getcwd()
    dir = os.path.join(cdir, 'server/models')
    filepath = os.path.join(dir, 'backloggd_games.csv')
    data = pd.read_csv(filepath)
    ratings = data[['Title', 'Genres', 'Platforms']]
    ratings = tf.data.Dataset.from_tensor_slices(ratings)
    ratings = ratings.map(lambda x: {
        "Title": x[0],
        "Genres": x[1],
    })
    games = ratings.map(lambda x: x["Title"])
    return games

def task():
    games = games_tensor()
    task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            games.batch(128).map(games_model)
        ),
    )
    return task



def recommend(request):
    # Get the current working directory
    cdir = os.getcwd()
    dir = os.path.join(cdir, 'server/models')
    games_model_filepath = os.path.join(dir, 'games_model')
    genres_model_filepath = os.path.join(dir, 'genres_model')

    games_model = tf.keras.models.load_model(games_model_filepath)
    genres_model = tf.keras.models.load_model(genres_model_filepath)

    # Construct the path to the model weights
    filepath = os.path.join(cdir, 'server/models/model_weights', 'model_weights')

    # Load the model weights
    model = GamesModel(games_model, genres_model, task)
    model.load_weights(filepath)

    index = tfrs.layers.factorized_top_k.BruteForce(model.genres_model)
    index.index_from_dataset(
        games_tensor().batch(4096).map(lambda title: (title, model.games_model(title))))

    # Get some recommendations.
    genre = request.GET.get('genre', "['Adventure', 'RPG']")
    _, titles = index(np.array([genre]))
    # print(f"Top 3 recommendations for Adventure, RPG and Turn Based Strategy: {list(set(list(np.array(titles[0, :10]))))[:3]}")
    game_list = list(set(list(np.array(titles[0, :30]))))[:20]
    response = {
        'games': str(game_list)
    }

    return JsonResponse(response)