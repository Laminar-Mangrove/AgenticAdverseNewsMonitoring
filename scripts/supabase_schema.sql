-- Run in Supabase SQL Editor (https://supabase.com/dashboard → SQL)
-- Enables email auth + per-user credits for the Adverse News Classifier app.

create table if not exists public.user_credits (
    user_id uuid primary key references auth.users (id) on delete cascade,
    email text not null,
    credits integer not null default 1 check (credits >= 0),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.stripe_sessions (
    session_id text primary key,
    user_id uuid not null references auth.users (id) on delete cascade,
    credits integer not null check (credits > 0),
    created_at timestamptz not null default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.user_credits (user_id, email, credits)
    values (new.id, new.email, 1)
    on conflict (user_id) do nothing;
    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();

create or replace function public.deduct_credit(p_user_id uuid)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    new_balance integer;
begin
    update public.user_credits
    set credits = credits - 1,
        updated_at = now()
    where user_id = p_user_id
      and credits >= 1
    returning credits into new_balance;

    if not found then
        return -1;
    end if;

    return new_balance;
end;
$$;

create or replace function public.add_credits(p_user_id uuid, p_amount integer)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    new_balance integer;
begin
    if p_amount <= 0 then
        raise exception 'p_amount must be positive';
    end if;

    update public.user_credits
    set credits = credits + p_amount,
        updated_at = now()
    where user_id = p_user_id
    returning credits into new_balance;

    if not found then
        raise exception 'user not found';
    end if;

    return new_balance;
end;
$$;

alter table public.user_credits enable row level security;
alter table public.stripe_sessions enable row level security;

create policy "Users can read own credits"
    on public.user_credits
    for select
    using (auth.uid() = user_id);

create policy "Users can read own stripe sessions"
    on public.stripe_sessions
    for select
    using (auth.uid() = user_id);
