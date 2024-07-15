from sklearn.linear_model import LassoCV


def generate_LASSO(min=-3, max=1):
    min, max = min, max
    alphas = np.logspace(min, max, 50)
    extractor = LassoCV(alphas=alphas, cv=5, max_iter=300000)
    return extractor

def generate_RELF():
    from skrebate import ReliefF
    return ReliefF()



