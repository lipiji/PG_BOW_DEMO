function string_out=num2string(num_in,string_length)

if(num_in~=0)
    eenheid = floor(log10(num_in))+1;
else
    eenheid = 1;
end

string_out='';
nul_string='0';
for ii=eenheid+1:string_length
    string_out=sprintf('%s%s',string_out,nul_string);
end
string_out=sprintf('%s%s',string_out,num2str(num_in));

