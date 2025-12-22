# Database Migrations
#  DÙNG SUPABASE CỦA T THÌ CHẮC KHÔNG CẦN CHẠY ĐÂU
These SQL migration files need to be executed in Supabase SQL Editor.

## How to Run Migrations

1. Go to your Supabase project dashboard
2. Navigate to **SQL Editor** (in the left sidebar)
3. Click **New Query**
4. Copy and paste the contents of the migration file
5. Click **Run** (or press Ctrl+Enter)

## Migration Order

Execute migrations in numerical order:

1. `001_create_parking_events.sql` - Creates the base parking_events table
2. `002_add_parking_fields.sql` - Adds license_plate, parking_slot, and event_type fields
3. `003_create_parking_slots.sql` - Creates parking_slots table to manage parking slots (run after 002_add_parking_fields.sql)
4. `003_insert_mock_data.sql` - **Optional**: Inserts comprehensive mock data for testing (20 events)
5. `004_insert_simple_mock_data.sql` - **Optional**: Inserts simple mock data for quick testing (5 scenarios)

**Note**: Run `003_create_parking_slots.sql` after `002_add_parking_fields.sql` to set up slot management.

## Mock Data
For testing the History page, you can run either:
- `003_insert_mock_data.sql` - Comprehensive test data with various scenarios
- `004_insert_simple_mock_data.sql` - Simple test data with 5 scenarios

**Note**: Mock data scripts are optional and can be run after setting up the schema.

## Alternative: Using Supabase CLI

If you have Supabase CLI installed, you can also run:

```bash
supabase db push
```

But you'll need to set up the Supabase project locally first.

