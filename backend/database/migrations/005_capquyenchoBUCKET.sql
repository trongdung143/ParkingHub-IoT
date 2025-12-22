
-- insert 
create policy "allow insert to parking-images"
on storage.objects
for insert
to public
with check (bucket_id = 'parking-images');

-- read from parking-images
create policy "allow read from parking-images"
on storage.objects
for select
to public
using (bucket_id = 'parking-images');