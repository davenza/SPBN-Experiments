from pgmpy.estimators.callbacks import DrawModel
from pgmpy.models import HybridContinuousModel

hcm = HybridContinuousModel.load_model('true_model.pkl')

drawer = DrawModel('./')
drawer.call(hcm, None, None, 1)