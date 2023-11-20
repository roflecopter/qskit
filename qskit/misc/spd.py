import datetime
def now():
    return(datetime.datetime.now())

def spd(s,ss):
    msg = f'{round((datetime.datetime.now()-ss).total_seconds(),1)}s | {round((datetime.datetime.now()-s).total_seconds(),1)}s run'
    return(msg)
