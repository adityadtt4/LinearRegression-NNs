# An analysis on the project

## The task

- The task involved using the Scikit-learn and pandas library and training a linear regression and multi layer perceptron regression model offered in order to try to predict the value of Premier League footballers based on their performance in the 2024/25 season

- The data was taken from a dataset available on Kaggle, which can be found through this link: https://www.kaggle.com/datasets/aesika/english-premier-league-player-stats-2425?resource=download

  - I used this dataset as it was, apart from four additional columns I added which was the transfer value of each player in Euros which were taken from Transfermarkt.com as well as additional columns denoting whether a player was an attacker, midfielder or defender
  - For most players, the transfer value used was their value on the 30th May 2025, which was 5 days after the end of the season. However, for players who didn't have a value assigned at this date, their transfer value was taken at most 3 months after this date



## Feature selection and preprocessing

- One decision I made was to remove all goalkeeper data and the associated goalkeeper statistic columns due to goalkeepers having drastically different stats being measured to compared to players in any other position on the pitch which could have produced noise for the models. Additionally, I felt comfortable about this decision since removing the position with the least amount of players in the dataset would have still allowed to me to have plenty of data entries available whilst not complicating the performance of the models

- In regard to preprocessing, there weren't any missing entries or incorrectly entered entries so I didn't have too much to do in this regard. However, given how large the range can be on some of the data, such as the transfer value for players, I did decide to incorporate MinMax scaling for all of the columns in the dataset.


## Results and final evaluation

- When splitting the dataset, I opted for a 75% training data and 25% testing data split. For the neural network model only, there were some additional settings I configured which were the maximum number of iterations which was set to 1000, the activation function used which was set to 'relu' as well as the hidden layer size which was set to 300.
  - These settings for the neural network was chosen as ReLU is one of the most popular activation functions for ANN's and the hidden layer size was chosen based on testing varying sizes and identifying a spot where the difference between residuals could be minimised to a more significant extent
- The two methods for evaluating performance for the models were the r^2 score as well as the mean squared error. Please find a table of results for these below:

| Model             | R² Score | Mean Squared Error (MSE) |
|-------------------|----------|--------------------------|
| Linear Regression | 0.344    | 0.011                    |
| MLP Regressor     | 0.450    | 0.009                    |

- As you can see, using a neural network improved the MSE by 18% which is a good improvement from the linear regression model. R^2 was also able to improve by 30%. This large performance difference isn't surprising given the power of ANN's and large amount of hidden layers used. However, none of these scores are particularly groundbreaking which takes me to my final point which relates to the limitations of the dataset.
- Firstly, there being only 500 or so players and there being 3 different player positions (attacker, midfielder and defender) which involve very different stats to inform the value of a player in x or y position means the models don't actually have much data to learn from. Additionally, the dataset is missing key features such as the age of each player which is a very significant factor in the value of a player.
- Finally, the reality is that the value of a footballer player is far more complicated than just their performance in a single season. For example the amount of years left at a club on the contract of a player or their prior injury history are also large factors that clubs consider when purchasing players. However, despite the limitations faced in this project, the models still managed to perform to an adequate standard.
