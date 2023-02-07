import statsmodels.api as sm

def build_OLS_data(data, model_order):

    # Sort each county by timepoint and take the first n (most recent) rows from each county
    recent_values_df = data.sort_values('first_day_of_month', ascending=False).groupby('cfips').head(model_order)

    # Sort by cfips, then timepoint
    recent_values_df.sort_values(['cfips', 'first_day_of_month'], inplace=True)

    # Number timepoints so we have an easy numeric x variable for regression
    if 'timepoint_num' not in list(data.columns):
        recent_values_df.insert(1, 'timepoint_num', recent_values_df.groupby(['cfips']).cumcount())

    # Clean up
    recent_values_df.reset_index(inplace=True, drop=True)

    return recent_values_df


def OLS_prediction(data, xinput, yinput, xforecast):

    model = sm.OLS(data[yinput], sm.add_constant(data[xinput])).fit()
    predictions = model.predict(sm.add_constant(xforecast))

    return predictions