Hi,

I need some help please. I've been working on this for 2 weeks solid. I've implemented generators, the CNN, got the testing loop working etc. I've been posting on Slack and the forums for over a week. I've tried everything suggested, and I still can;t get this to work.

I think the issue is due to the keyboard steering training data giving 'bad' data. Because you tap on the turn keys, you end up with value sequences like -0.8, 0.0, 0.0, -0.4, 0.0, 0.0 etc when steering left. I've tried various smoothing strategies (see in code). I've also tried the older 50Hz simulator. But nothing is giving me good values. The car just slowly steers off to the left and hits the kerb. 

I'm training with about 20k samples. As AWS have not enabled me to run on GPU yet (waiting approval), I'm running on my CPU. It's taking between 4-12 hours to train. So it's making 'what if's' very frustrating.

I'm at the point where I need some guidance please. I'm feeling like the excercise has turned into one of 'fix the bad data from simulator' rather than create a behavioral simulator. 

Could you please take a look at my model and see if it's ok? And any pointers as to how I can get this working would be appreciated.

Thanks,
Paul
# BehaviouralCloning
