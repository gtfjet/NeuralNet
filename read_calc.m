clear,clc,close all


fid = fopen('out.bin','r');
f = fread(fid,[22 3335],'single')';
fclose(fid);

fid = fopen('calc.bin','r');
y = fread(fid,[22 3335],'single')';
fclose(fid);

for i=1:22
    plot(f(:,i),y(:,i),'o')
    hold on
end