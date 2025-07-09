% before debugging anything, check to make sure the well ports are lined up.  right now the code uses:
% ports : 1 2 3 5 6 7 8 9 11, if this is incorrect, you need to change it before proceeding


%VARIABLES
int deliverPeriod1 = 500		% reward duration- adjust this based on pump
int loopInterval=1 				% just do very fast (1 msec run randomizer again)

int rewardWell= 4      			% reward well (pump number)
int nextWell= 4					% this is the well of the future reward (1:3)
int currWell = 4					% well theyre both currently at (1:3)
int lastWell = 0					% just to make sure we dont keep printing at the same well
int count= 0             		   		% reward count (increment after reward and post)
int trycount=0 					% try count

int rewardPump1 = 3 			% for wells 1 and A
int rewardPump2 = 5 			% for wells 2 and B
int rewardPump3 = 1			% for wells 3 and C

% to keep track of whether both rats broke beams
int wellStat1=0 					% well 1 is 3
int wellStat2=0  					% well 2 is 2
int wellStat3=0 					% well 3 is 8
% int wellStatA=0 					% well A is 4
% int wellStatB=0 					% well B is 9
% int wellStatC=0 					% well C is 11

% Keep 	tra ck of the total number of transitions for each rat 

int rat1_transitions = 0

int rat1_previouswell = 0


updates off 16						% this is the camera strobe


% Reward delivery function (basically we have two animals at matching ports)
function 1
	
	portout[rewardWell]=1 					% reward
	do in deliverPeriod1 						% do after waiting deliverPeriod milliseconds
		portout[rewardWell]=0 				% reset reward
	end

	nextWell=random(2)+1						% basically if the next well isnt different from this well, keep trying
	while (nextWell == currWell) do every loopInterval
		nextWell = random(2)+1
	then do

		disp(nextWell)
	end

	
	count = count+1
	disp(count)	
		
	% trycount=trycount+1
	% disp('match number')
	% disp(trycount)
end;




%
% beam break functions
%

% well 1
callback portin[2] up
	disp('Poke in well1') 						% Print state of port to terminal
	wellStat1 = 1
	if (rat1_previouswell!=1) do
		rat1_transitions=rat1_transitions+1
		disp(rat1_transitions)
		rat1_previouswell=1
	end
								% Check if we have a match
	currWell = 1
	if ((rat1_previouswell != currWell) & (currWell==nextWell))do			% check if this is the first match

			
		disp('Rewarding well1 ')
		lastWell = 1						% this is the well we're at
		rewardWell = rewardPump1 	% dispense reward from here		
		trigger(1) % trigger reward
		
	end
end;

callback portin[2] down
	disp('UnPoke in well1') 				% Print state of port to terminal
	wellStat1=0
end;


% well 2
callback portin[10] up
	disp('Poke in well2') 					% Print state of port to terminal
	wellStat2 = 1
	if (rat1_previouswell!=2) do
		rat1_transitions=rat1_transitions+1
		disp(rat1_transitions)
		rat1_previouswell=2
	end
					% Check if previous well = center
	currWell = 2
	if (lastWell != currWell & currWell==nextWell) do
			
			
		disp('Rewarding well2 ')
		lastWell = 2						% this is the well we're at
		rewardWell = rewardPump2 	% dispense reward from here		
		trigger(1)	
		
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
	if (rat1_previouswell!=3) do
		rat1_transitions=rat1_transitions+1
		disp(rat1_transitions)
		rat1_previouswell=3
	end
					% Check if matched well
	currWell = 3
	if (lastWell != currWell & currWell==nextWell) do			% check if this is first match this trial
			
		
		disp('Rewarding well3')
		lastWell = 3
		rewardWell = rewardPump3 	% dispense reward from here	
		trigger(1)							% trigger reward
		
	end
end;

callback portin[13] down
	disp('UnPoke in well3') 	% Print state of port to terminal
	wellStat3 = 0
end;



