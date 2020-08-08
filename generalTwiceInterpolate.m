% Do nearest interpolation of sub-sampled sinogram
function generalTwiceInterpolate(cload,csave,from,to)
for i = from:to
    fname = strcat('./TestData/',string(cload),string(i),'.mat');
    load(fname);
    I = Rref;
    sz = size(I);
    Rref = imresize(I,[2*sz(1),sz(2)],'nearest');
    fsave = strcat('./TestData/',string(csave),string(i),'.mat');
    save(fsave,'Rref');
end
end
