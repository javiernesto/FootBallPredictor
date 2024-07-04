import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    # Loading the data
    df_matches = pd.read_csv('DataSource/results.csv')
    df_nations = pd.read_csv('DataSource/CopaAmerica2024.csv')
    df_ranking_fifa = pd.read_csv('DataSource/RankingFIFA.csv')

    # Transforming the data
    df_nations_ranking = pd.merge(df_nations, df_ranking_fifa, on='Nation', how='left')

    df_matches['date'] = pd.to_datetime(df_matches['date'], format="%Y-%m-%d")
    df_matches = pd.merge(df_matches, df_ranking_fifa, left_on='home_team', right_on='Nation', how='left')
    df_matches = pd.merge(df_matches, df_ranking_fifa, left_on='away_team', right_on='Nation', how='left')


    df_matches_hist = df_matches[(df_matches['date'] > '2021-01-01') & (df_matches['date'] < '2024-06-13')]
    df_matches_tournament_home = pd.merge(df_matches_hist, df_nations_ranking, left_on='home_team', right_on='Nation',
                                    how='inner')
    df_matches_tournament_away = pd.merge(df_matches_hist, df_nations_ranking, left_on='away_team', right_on='Nation',
                                    how='inner')

    df_matches_hist = pd.concat([df_matches_tournament_home, df_matches_tournament_away], ignore_index=True)
    df_matches_hist = df_matches_hist[
        ['Rank_x', 'Points_x', 'Difference_x', 'Rank_y', 'Points_y', 'Difference_y', 'home_score', 'away_score']]
    df_matches_hist['Difference_x'] = df_matches_hist['Difference_x'].fillna(0)
    df_matches_hist['Difference_y'] = df_matches_hist['Difference_y'].fillna(0)
    df_matches_hist['Difference_x'] = df_matches_hist['Difference_x'].map({'-': 0})
    df_matches_hist['Difference_y'] = df_matches_hist['Difference_y'].map({'-': 0})

    df_group_stage_matches = df_matches[
        (df_matches['tournament'] == 'Copa America') & (df_matches['country'] == 'USA') & (
                df_matches['date'] > '2024-06-19') & (df_matches['date'] < '2024-07-04')].copy()

    df_group_stage_matches['Difference_x'] = df_group_stage_matches['Difference_x'].fillna(0)
    df_group_stage_matches['Difference_y'] = df_group_stage_matches['Difference_y'].fillna(0)
    df_group_stage_matches['Difference_x'] = df_group_stage_matches['Difference_x'].map({'-': 0})
    df_group_stage_matches['Difference_y'] = df_group_stage_matches['Difference_y'].map({'-': 0})

    df_x = df_matches_hist[['Rank_x', 'Points_x', 'Difference_x', 'Rank_y', 'Points_y', 'Difference_y']].to_numpy()
    df_y_home = df_matches_hist['home_score'].to_numpy()
    df_y_away = df_matches_hist['away_score'].to_numpy()
    home_scores = predict_score(df_group_stage_matches, df_x, df_y_home)
    away_scores = predict_score(df_group_stage_matches, df_x, df_y_away)

    df_group_stage_matches['home_score'] = pd.Series(home_scores, index=df_group_stage_matches.index)
    df_group_stage_matches['away_score'] = pd.Series(away_scores, index=df_group_stage_matches.index)
    df_group_stage_matches.to_csv('prediction.csv')


def predict_score(df_matches_to_predict, df_x, df_y):
    x_train1, x_test1, y_train1, y_test1 = train_test_split(df_x, df_y, test_size=0.30)
    model = DecisionTreeClassifier()
    model.fit(x_train1, y_train1)
    scores = model.predict(
        df_matches_to_predict[['Rank_x', 'Points_x', 'Difference_x', 'Rank_y', 'Points_y', 'Difference_y']].to_numpy())
    return scores


if __name__ == '__main__':
    main()
