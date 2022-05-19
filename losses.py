def diceLoss(pred,target,smooth =1e-3):
    """
    parameters:
                -predication 
                -target
    retruns: -loss as 1-dice
             -metric as dice
    """
    intersection = (pred *target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3))+target.sum(dim=(2,3))
    dice = 2*(intersection)/(union+smooth)#Metric
    diceloss = 1-dice#sum
    return diceloss.sum(),dice.mean()

