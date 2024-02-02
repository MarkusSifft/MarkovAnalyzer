# This file is part of MarkovAnalyzer
#
#    Copyright (c) 2020 and later, Markus Sifft and Daniel Hägele.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

from scipy.stats import chi2


def compare_models(result1, result2):
    """
    Compare two lmfit.MinimizerResult objects.

    Parameters:
    - result1: lmfit.MinimizerResult for the simpler model
    - result2: lmfit.MinimizerResult for the complex model

    Returns:
    - A dictionary with AIC, BIC for each model and the p-value for the Likelihood Ratio Test
    """

    AIC1 = result1.aic
    BIC1 = result1.bic

    AIC2 = result2.aic
    BIC2 = result2.bic

    # Extract the log likelihoods from the fit results
    log_likelihood_model1 = -0.5 * result1.chisqr
    log_likelihood_model2 = -0.5 * result2.chisqr

    # Calculate the test statistic
    test_statistic = -2 * (log_likelihood_model1 - log_likelihood_model2)

    # Calculate the degrees of freedom, which is the difference in the number of parameters
    df = len(result2.params) - len(result1.params)

    # Calculate the p-value
    p_value = chi2.sf(test_statistic, df)

    # Advice section
    advice = "Model Comparison Advice:\n"

    if AIC1 < AIC2 and BIC1 < BIC2:
        advice += "Both AIC and BIC suggest the simpler model is better.\n"
    elif AIC2 < AIC1 and BIC2 < BIC1:
        advice += "Both AIC and BIC suggest the complex model is better.\n"
    else:
        advice += "AIC and BIC disagree, consider other model validation techniques.\n"

    if p_value < 0.05:
        advice += "The Likelihood Ratio Test suggests rejecting the simpler model in favor of the complex model."
    else:
        advice += "The Likelihood Ratio Test does not suggest rejecting the simpler model."

    results = {
        'AIC1': AIC1, 'AIC2': AIC2,
        'BIC1': BIC1, 'BIC2': BIC2,
        'Likelihood Ratio Test p-value': p_value,
        'Advice': advice
    }

    print(advice)

    return results