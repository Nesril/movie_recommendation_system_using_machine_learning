import pandas as pd

data = pd.read_csv('MovieRecommendations.csv')
#inputMovieName = input("input movie name: ")

def getRecommendations(movie):
    list_result = data[data['title'] == movie]
    fm = list_result['FirstMovieRecommendation'].values[0]
    sm = list_result['SecondMovieRecommendation'].values[0]
    tm = list_result['ThirdMovieRecommendation'].values[0]
    fourthm = list_result['FourthMovieRecommendation'].values[0]
    finalRecommendationText = '1:' + fm + ' \n2:' + sm + ' \n3:' + tm + ' \n4:' + fourthm
    print('Your Recommendations for the Movie ' + movie + ' are:\n')
    print(finalRecommendationText)
getRecommendations("101 Dalmatians (1996") 