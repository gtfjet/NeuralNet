clear,clc,close all

% Name of results file
dsres = 'TestFSM.mat';

% List of states
states={'FSM.flowMeter.volume.p'
        'FSM.IFPC.EBV.dampingCavity.s'
        'FSM.IFPC.EBV.spoolValve.p'
        'FSM.IFPC.EBV.volume.p'
        'FSM.IFPC.FDV.dampingCavity.s'
        'FSM.IFPC.MPSOV.cavity.s'
        'FSM.IFPC.MPSOV.solenoid.x'
        'FSM.IFPC.MV.cavity.s'
        'FSM.IFPC.MV.flapper.x'
        'FSM.IFPC.MV.pistonP1.p'
        'FSM.IFPC.MV.volumeP2.p'
        'FSM.IFPC.MV.volumePSO.p'
        'FSM.IFPC.PES.flowArea.x'
        'FSM.IFPC.PRV.dampingCavity.s'
        'FSM.IFPC.pumpBoost.dp'
        'FSM.IFPC.volumeSRS.p'
        'FSM.IFPC.WMBV.dampingCavity.s'
        'FSM.interstage.volumeFF.p'
        'FSM.interstage.volumeFOC.p'
        'FSM.nozzles.volumePri.p'
        'FSM.nozzles.volumeSec.p'
        'FSM.speed.phi'};
    
inputs={'N2.y','PB.y','PIN.y','itm.y'};
        
nx = length(states);
nu = length(inputs);
d = dymload(dsres);
t = dymget(d,'Time');
n = length(t);
x = single(zeros(n,nx+nu));
f = single(zeros(n,nx));

for i = 1:nx
    [tok, rem] = strtok(fliplr(states{i}),'.');
    x(:,i) = normalize(dymget(d,states{i}));
    f(:,i) = normalize(dymget(d,[fliplr(rem) 'der(' fliplr(tok) ')']));
end

for i = 1:nu
    x(:,nx+i) = normalize(dymget(d,inputs{i}));
end

fid = fopen('in.bin','w');
fwrite(fid,x','single');
fclose(fid);

fid = fopen('out.bin','w');
fwrite(fid,f','single');
fclose(fid);

disp(nx+nu);
disp(nx);
disp(n);



