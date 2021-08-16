X_payload_uint8=cell(size(X_payload));
for i=1:1:size(X_payload,2)
    X_payload_uint8{i}=uint8(X_payload{i});
end
    