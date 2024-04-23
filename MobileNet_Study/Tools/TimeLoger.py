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
            lastMess = "{} spend {} seconds\n{}\n".format(missionName, t, '='*30)
            totalMess = "{}\n{}".format(logMess,lastMess)
            log.write(totalMess)
            log.flush()
            log.close()
            print(lastMess)
        return forward
