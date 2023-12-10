import time
class TimeLogMission:
    @staticmethod
    def timeLogMission(f):
        def forward(missionName, logPath, *args,**kwargs):
            time_1 = time.time()
            log = open(logPath,'a+')
            logMess = f(*args,**kwargs)
            time_2 = time.time()
            t = time_2 - time_1
            totalMess = "{}\n{} spend {} seconds\n{}\n".format(logMess,missionName,t,'='*30)
            log.write(totalMess)
            log.flush()
            log.close()
            print(totalMess)
        return forward
