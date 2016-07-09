function y = normalize(u)
    y = (u-min(u))/max(eps, max(u)-min(u));
end





