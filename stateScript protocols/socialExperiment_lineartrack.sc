% synchronization maze training
% 4 pumps, two linear tracks, the animals have to poke at the same time at the same well to get reward,
% then they have to do the same at the other well.  This task is deterministic, so they can do this without
% regarding their partner




int deliverPeriod1 = 500 % reward duration- adjust this based on pump


%VARIABLES


% vars for tracking behavior in maze


int lastWell = 4		 	% 1 if left, 3 if right, 2 if center ... this variable trackes the previously activated well
% vars for tracking where and how much reward
int rewardWell1 = 0       	% reward well
int rewardWellA = 0       	% reward well
int nowRewarding = 0 	% variable that keeps tabs on the reward being dispensed .. when reward is being dispensed and the system is in the midst of executing a reward function, this number hops up to a 1, and then relaxes to 0 when reward is finished.


int count = 0                	% blink count


% these are the current well IR and PUMP pins:
int rewardPump1 = 3 	% well PUMP 1 port 3
int rewardPump2 = 1 	% well PUMP 2 port 1
int rewardPumpA = 5 	% well PUMP A port 5
int rewardPumpB = 8	% well PUMP B port 8




int wellStat1=0 % well 1 is 3
int wellStat2=0  % well 2 is 2
%int wellStat3=0 % well 3 is 8
int wellStatA=0 % well 4 is 1
int wellStatB=0 % well 5 is 9
%int wellStatC=0 % well 6 is 11


int rewardNumber = 0


updates off 16


% Reward delivery function
function 1
	nowRewarding = 1 							% nowRewarding
	disp('now rewarding')	
	portout[3]=1 					% reward
 	portout[5]=1
	do in deliverPeriod1 						% do after waiting deliverPeriod milliseconds
		portout[3]=0 				% reset reward
		portout[5]=0 				% reset reward
		nowRewarding=0 					% no longer rewarding
	end
end;


% Reward delivery function
function 2
	nowRewarding = 1 							% nowRewarding
	disp('now rewarding')	
	portout[1]=1 					% reward
 	portout[8]=1
	do in deliverPeriod1 						% do after waiting deliverPeriod milliseconds
		portout[1]=0 				% reset reward
		portout[8]=0 				% reset reward
		nowRewarding=0 					% no longer rewarding
	end
end;




% beam break functions
% well 1
callback portin[15] up
	disp('Poke in well1') 		% Print state of port to terminal
	rewardWell1=rewardPump1 	% dispense reward from here
	wellStat1=1
	if (lastWell !=1 && wellStatA==1) 	do				% Check if previous well = center
			disp('Rewarding - position A1 ')
			lastWell=1		
			trigger(1)					% trigger reward
			rewardNumber = rewardNumber + 1
			disp(rewardNumber)	
	end
end;




callback portin[15] down
	disp('UnPoke in well1') 	% Print state of port to terminal
	wellStat1=0
end;




% well 2
callback portin[10] up
	disp('Poke in well2') 		% Print state of port to terminal
	wellStat2=1
	rewardWell1=rewardPump2 	% dispense reward from here
	if (lastWell !=2 && wellStatB==1) 	do				% Check if previous well = center
			disp('Rewarding position B2 ')
			
			lastWell=2						% trigger reward
			trigger(2)
			rewardNumber = rewardNumber + 1
			disp(rewardNumber)	
	end
end;


callback portin[10] down
	disp('UnPoke in well2') 	% Print state of port to terminal
	wellStat2=0
end;






% well A
callback portin[2] up
	disp('Poke in wellA') 		% Print state of port to terminal
	wellStatA=1
	rewardWellA=rewardPumpA 	% dispense reward from here
	if (lastWell !=1 && wellStat1==1) do					% Check if previous well = center
			disp('Rewarding position A1 ')
			lastWell=1		
			trigger(1)					% trigger reward
			rewardNumber = rewardNumber + 1
			disp(rewardNumber)	
	end
end;




callback portin[2] down
	disp('UnPoke in wellA') 	% Print state of port to terminal
	wellStatA=0
end;


% well b
callback portin[4] up
	disp('Poke in wellB') 		% Print state of port to terminal
	wellStatB=1
	rewardWellA=rewardPumpB 	% dispense reward from here
	if (lastWell !=2 && wellStat2==1) 	do				% Check if previous well = center
			disp('Rewarding position B2 ')
			lastWell=2		
			trigger(2)					% trigger reward
			rewardNumber = rewardNumber + 1
			disp(rewardNumber)	
	end
end;




callback portin[4] down
	disp('UnPoke in wellB') 	% Print state of port to terminal
	wellStatB=0
end;




