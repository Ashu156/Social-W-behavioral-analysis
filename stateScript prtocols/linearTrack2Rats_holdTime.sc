int deliverPeriod1 = 600 % reward duration- adjust this based on pump (milliseconds)
int holdTime = 200 % how long they need to hold (milliseconds)
%VARIABLES


% vars for tracking behavior in maze
int rewardWell12 = 1 % the pump to actuate (will be assigned rewardPump)
int rewardWellAB = 1


int lastWell12= 3 % last rewarded well
int lastWellAB= 3


int count12 = 0
int countAB = 0               	% reward count


int lockout12=0
int lockoutAB=0


int currWell12=0 % to keep track of whether its still broken
int currWellAB=0


int dummytrue=1


% these are the current well IR and PUMP pins:
int rewardPump1 = 3 % well PUMP 1 port 3
int rewardPump2 = 1 % well PUMP 2 port 1
int rewardPumpA = 5 	% well PUMP A port 5
int rewardPumpB = 8 % well PUMP B port 8



updates off 16


% Reward delivery function
function 1
	portout[rewardWell12] = 1 					% reward
	do in deliverPeriod1 					% do after waiting deliverPeriod milliseconds
		portout[rewardWell12] = 0 				% reset reward
	end
end;


function 2
portout[rewardWellAB] = 1 					% reward
	do in deliverPeriod1 					% do after waiting deliverPeriod milliseconds
		portout[rewardWellAB] = 0 				% reset reward
	end
end;
% beam break functions


% well 1 (port 9)


callback portin[15] up
	disp('Poke in well1') 		% Print state of port to terminal
	currWell12 = 1
	if (lockout12==0) do
		lockout12 = 1
		if (lastWell12 !=1) 	do in holdTime	% Check if previous well = center
			if (currWell12==1) do
				disp('Rewarding - position 1 rewards:')
				rewardWell12=rewardPump1 	% dispense reward from here
				lastWell12=1		
				trigger(1) % trigger reward
				count12 = count12 + 1
				disp(count12)
			end
		end
		if (dummytrue==1) do in holdTime
			lockout12=0
		end
	end
end;




callback portin[15] down
	disp('UnPoke in well1') 	% Print state of port to terminal
	currWell12=0
end;


% well 2 (port 11)


callback portin[10] up
	disp('Poke in well2') 		% Print state of port to terminal
	currWell12=2
	if (lockout12==0) do
		lockout12=1
		if (lastWell12 !=2)	do	in holdTime	
			if (currWell12==2) do
				disp('Rewarding position 2 rewards:')
				rewardWell12=rewardPump2 	% dispense reward from here
				lastWell12=2		
				trigger(1)
				count12 = count12 + 1
				disp(count12)	
			end
		end
		if (dummytrue==1) do in holdTime
			lockout12=0
		end
	end
end;


callback portin[10] down
	disp('UnPoke in well2') 	% Print state of port to terminal
	currWell12=0
end;


% Well A (now port 1)


callback portin[1] up
	disp('Poke in wellA') 		% Print state of port to terminal
	currWellAB=1
	if (lockoutAB==0) do
		lockoutAB=1
		if (lastWellAB !=1) do in holdTime	% Check if previous well = center		
			if (currWellAB==1) do
				disp('Rewarding position A rewards:')
				rewardWellAB=rewardPumpA 	% dispense reward from here
				lastWellAB=1	
				trigger(2)
				countAB = countAB + 1
				disp(countAB)
			end
		end
		if (dummytrue==1) do in holdTime
			lockoutAB=0
		end
	end
end;


callback portin[1] down
	disp('UnPoke in wellA') 	% Print state of port to terminal
	currWellAB=0
end;


% well B (port 3)


callback portin[4] up
	disp('Poke in wellB') 		% Print state of port to terminal
	currWellAB=2
	if (lockoutAB==0) do
		lockoutAB=1
		if (lastWellAB !=2) do in holdTime				% Check if previous well = center
			if (currWellAB==2) do
				disp('Rewarding position B rewards:')
				rewardWellAB=rewardPumpB 	% dispense reward from here
				lastWellAB=2				
				trigger(2)					% trigger reward
				countAB = countAB + 1
				disp(countAB)
			end
		end
		if (dummytrue==1) do in holdTime
			lockoutAB=0
		end
	end
end;




callback portin[4] down
	disp('UnPoke in wellB') 	% Print state of port to terminal
	currWellAB=0
end;