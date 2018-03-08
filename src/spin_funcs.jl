function pauliX(vec::Int, pos::Int64)
    mask = 1<<(pos-1)
    return 1, xor(vec-1, mask)+1
end
function pauliY(vec::Int64, pos::Int64)
    mask = 1<<(pos-1)
    if ((vec-1) & mask) == 0
        sign = -1im
    else
        sign = 1im
    end
    return sign, xor(vec-1,mask)+1
end
function pauliZ(vec::Int64, pos::Int64)
    mask = 1<<(pos-1)
    if  ((vec-1) & mask) == 0
        sign = -1
    else
        sign = 1
    end
    return sign, vec
end

const pauli = [pauliX,pauliY,pauliZ]

function changeBit(vector::Int64, spin::Int64)
    v = vector - 1
    s = spin - 1
    return xor(v,(1<<s))+1
end
function invert(vector::Int64, spin1::Int64, spin2::Int64)
    v = vector - 1
    s1 = spin1 - 1
    s2 = spin2 - 1
    return xor(v,((1<<s1) + (1<<s2)))+1
end
function getSign(vector::Int64, spin1::Int64, spin2::Int64)
    v = vector - 1
    s1 = spin1 - 1
    s2 = spin2 - 1
    bit1 = (v >> s1) & 1
    bit2 = (v >> s2) & 1
    return 1 - (xor(bit1,bit2)<<1)
end
