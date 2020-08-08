% Make patches from existing sinogram data.
% Save patches of difference of interpolated and original sinogram.
n = 0;
count = 0;
S = 32;
for i = 3:3:14040
    fname = strcat('./Data_Generated_clean/Rref',string(i),'.mat');
    load(fname);
    if(sum(isnan(Rref(:))) == 0)
        n = mod(n,3) + 1;
        B = Rref(1:2:100,:);
        [Bn,snr] = addNoise(B, n*20, 'peak');
        C = imresize(Bn,[100,512],'nearest');
        Df = C - Rref;
        for j = 1:50
            count = count + 1;
            disp(count);
            y = randi(100 - S + 1);
            x = randi(512 - S + 1);
            P = C(y:y+S-1,x:x+S-1);
            D = Df(y:y+S-1,x:x+S-1);
            fsave = strcat('./SinoPatches/P',string(count),'.mat');
            save(fsave,'P');
            fsave = strcat('./SinoPatches/D',string(count),'.mat');
            save(fsave,'D');
        end
    end
end
