% before debugging anything, check to make sure the well ports are lined up.  right now the code uses:
% ports : 1 2 3 5 6 7 8 9 11, if this is incorrect, you need to change it before proceeding




%VARIABLES
int deliverPeriod1 =1400 		% reward duration- adjust this based on pump
int loopInterval=1 				% just do very fast (1 msec run randomizer again)


int rewardWell= 4      			% reward well (pump number)
int nextWell= 4					% this is the well of the future reward (1:3)
int currWell = 4					% well theyre both currently at (1:3)
int lastWell = 0					% just to make sure we dont keep printing at the same well
int count= 0             		   		% reward count (increment after reward and post)
int trycount=0 					% try count


int rewardPump1 = 3			% for wells 1 and A
int rewardPump2 = 5 			% for wells 2 and B
int rewardPump3 = 1			% for wells 3 and C


% to keep track of whether both rats broke beams
int wellStat1=0 					% well 1 is 3
int wellStat2=0  					% well 2 is 2
int wellStat3=0 					% well 3 is 8
int wellStatA=0 					% well A is 4
int wellStatB=0 					% well B is 9
int wellStatC=0 					% well C is 11


updates off 16						% this is the camera strobe




% Reward delivery function (basically we have two animals at matching ports)
function 1
	if ((nextWell == 4) ||  (nextWell == currWell)) do
		portout[rewardWell]=1 					% reward
		do in deliverPeriod1 						% do after waiting deliverPeriod milliseconds
			portout[rewardWell]=0 				% reset reward
		end


	
		count = count+1
		disp(count)	
		nextWell=random(2)+1						% basically if the next well isnt different from this well, keep trying
		while (nextWell == currWell) do every loopInterval
			nextWell = random(2)+1
		then do


			disp(nextWell)
		end
	else do
		disp('but, no reward')	
	end
	trycount=trycount+1
	disp('match number')
	disp(trycount)
end;




%
% beam break functions
%


% well 1
callback portin[15] up
	disp('Poke in well1') 						% Print state of port to terminal
	wellStat1 = 1
	if (wellStatA == 1) do							% Check if we have a match
		currWell = 1
		if (lastWell != currWell) do			% check if this is the first match
			disp('Matched Pokes in position A1 ')
			lastWell = 1						% this is the well we're at
			rewardWell = rewardPump1 	% dispense reward from here		
			trigger(1) % trigger reward
		end
	end
end;


callback portin[15] down
	disp('UnPoke in well1') 				% Print state of port to terminal
	wellStat1=0
end;




% well 2
callback portin[10] up
	disp('Poke in well2') 					% Print state of port to terminal
	wellStat2 = 1
	if (wellStatB == 1)	do					% Check if previous well = center
		currWell = 2
		if (lastWell != currWell) do
			disp('Matched pokes in position B2 ')
			lastWell = 2						% this is the well we're at
			rewardWell = rewardPump2 	% dispense reward from here		
			trigger(1)	
		end
	end
end;


callback portin[10] down
	disp('UnPoke in well2') 				% Print state of port to terminal
	wellStat2 = 0
end;




% Well 3
callback portin[13] up
	disp('Poke in well3') 					% Print state of port to terminal
	wellStat3 = 1
	if (wellStatC == 1) 	do					% Check if matched well
		currWell = 3
		if (lastWell != currWell) do			% check if this is first match this trial
			disp('Matched Pokes in position C3 ')
			lastWell = 3
			rewardWell = rewardPump3 	% dispense reward from here	
			trigger(1)							% trigger reward
		end
	end
end;


callback portin[13] down
	disp('UnPoke in well3') 	% Print state of port to terminal
	wellStat3 = 0
end;




% well A
callback portin[2] up
	disp('Poke in wellA') 		% Print state of port to terminal
	wellStatA = 1
	if (wellStat1 == 1) do					% Check if previous well = center
		currWell = 1
		if (lastWell != currWell) do
			disp('Matched poke in position A1 ')
			lastWell = 1
			rewardWell = rewardPump1 	% dispense reward from here		
			trigger(1)					% trigger reward
		end
	end
end;




callback portin[2] down
	disp('UnPoke in wellA') 	% Print state of port to terminal
	wellStatA = 0
end;




% well B
callback portin[4] up
	disp('Poke in wellB') 		% Print state of port to terminal
	wellStatB = 1
	if (wellStat2==1) 	do				% Check if previous well = center
		currWell = 2
		if (currWell != lastWell) do
			disp('Matched Pokes in position B2 ')
			rewardWell = rewardPump2 	% dispense reward from here
			lastWell = 2		
			trigger(1)					% trigger reward
		end
	end
end;




callback portin[4] down
	disp('UnPoke in wellB') 	% Print state of port to terminal
	wellStatB=0
end;


% well C
callback portin[8] up
	disp('Poke in wellC') 		% Print state of port to terminal
	wellStatC=1
	if (wellStat3 == 1) 	do				% Check if previous well = center
		currWell = 3
		if (currWell != lastWell) do
			disp('Matched Pokes in position C3 ')
			rewardWell = rewardPump3	% dispense reward from here
			lastWell = 3		
			trigger(1)					% trigger reward
		end
	end
end;


callback portin[8] down
	disp('UnPoke in wellC') 	% Print state of port to terminal
	wellStatC=0
end;


